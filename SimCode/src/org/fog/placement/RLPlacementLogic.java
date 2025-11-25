package org.fog.placement;

import org.apache.commons.math3.util.Pair;
import org.fog.application.AppEdge;
import org.fog.application.AppModule;
import org.fog.application.Application;
import org.fog.entities.FogDevice;
import org.fog.entities.PlacementRequest;
import org.fog.entities.Tuple;
import org.fog.utils.ModuleLaunchConfig;
import org.fog.utils.Logger;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import com.google.gson.reflect.TypeToken;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.Executors;

/**
 * 强化学习部署逻辑类
 * 负责与 Python Agent 交互，执行部署动作，并反馈状态和奖励。
 * 包含环境随机化逻辑以模拟真实动态场景。
 */
public class RLPlacementLogic implements MicroservicePlacementLogic {

    // --- [配置] 必须与 Python config.py 一致 ---
    private static final int MAX_NODES = 50;
    private static final int API_PORT = 4567;

    // --- 成员变量 ---
    private List<FogDevice> fogDevices; // 所有设备引用
    private List<FogDevice> deployableNodes; // 仅包含 Cloud, Gateway, Edge
    private Map<Integer, FogDevice> fogDeviceMap;
    private Map<String, Application> applicationInfo;
    private List<PlacementRequest> placementRequests; // 保存原始请求供Reset使用

    // 内部状态 (Episode 相关)
    private LinkedList<QueuedModule> placementQueue; // 待部署模块队列
    private Map<String, Integer> currentPlacementMap; // 已部署结果
    private Map<Integer, Double> currentCpuLoad; // 当前模拟 CPU 负载
    private Map<Integer, Integer> currentRamLoad; // 当前模拟 RAM 负载
    private int currentModuleIndex = 0;

    // 服务器组件
    private HttpServer server;
    private Gson gson = new Gson();
    private static volatile boolean serverRunning = false; // 简单的状态标记

    // --- 辅助类定义 ---
    private static class QueuedModule {
        String moduleName, appId;
        AppModule moduleObj;
        public QueuedModule(String m, String a, AppModule o) { moduleName=m; appId=a; moduleObj=o; }
        public String getKey() { return appId + "_" + moduleName; }
    }

    static class StateRepresentation {
        List<Double> stateVector;
        List<Boolean> actionMask;
        String description;
        StateRepresentation(List<Double> s, List<Boolean> m, String d) { stateVector=s; actionMask=m; description=d; }
    }

    static class ActionResult {
        StateRepresentation nextStateRepresentation;
        double immediateReward;
        boolean done;
        ActionResult(StateRepresentation s, double r, boolean d) { nextStateRepresentation=s; immediateReward=r; done=d; }
    }

    static class FinalResult { double finalReward; FinalResult(double r) { finalReward=r; } }
    static class ErrorResponse { String message; ErrorResponse(String msg) { message = msg; } }

    // 构造函数
    public RLPlacementLogic(int fonId) {}

    @Override
    public PlacementLogicOutput run(List<FogDevice> fogDevices, Map<String, Application> applicationInfo,
                                    Map<Integer, Map<String, Double>> resourceAvailability, List<PlacementRequest> pr) {
        this.fogDevices = fogDevices;
        this.applicationInfo = applicationInfo;
        this.placementRequests = pr;

        // 建立 ID 索引
        this.fogDeviceMap = new HashMap<>();
        for (FogDevice d : fogDevices) fogDeviceMap.put(d.getId(), d);

        // 筛选可部署节点 (Level <= 2, 排除传感器和执行器)
        this.deployableNodes = new ArrayList<>();
        for (FogDevice dev : fogDevices) {
            if (dev.getLevel() <= 2) deployableNodes.add(dev);
        }
        // 排序以保证 State 向量中的节点顺序固定
        deployableNodes.sort(Comparator.comparingInt(FogDevice::getId));

        System.out.println("RL Logic initialized. Deployable nodes: " + deployableNodes.size());
        System.out.println("Waiting for Python Agent to connect...");

        // 启动服务器
        startRestApiServerOnce();

        // 阻塞主线程，等待 RL 训练完成 (由 StopHandler 唤醒)
        synchronized(this) {
            try { this.wait(); } catch (InterruptedException e) { e.printStackTrace(); }
        }

        System.out.println("RL Training finished. Generating final placement for iFogSim...");
        return generateFinalOutput();
    }

    // --- [核心逻辑] 重置环境 (引入随机性) ---
    private void resetInternalState(List<PlacementRequest> requests) {
        this.placementQueue = new LinkedList<>();
        this.currentPlacementMap = new HashMap<>();
        this.currentCpuLoad = new HashMap<>();
        this.currentRamLoad = new HashMap<>();
        this.currentModuleIndex = 0;

        // --- 随机背景负载，打破静态环境 ---
        Random rand = new Random();
        for (FogDevice dev : deployableNodes) {
            double totalMips = dev.getHost().getTotalMips();
            double loadFactor = 0.0;

            // 随机生成 0% ~ 90% 的背景负载，模拟真实场景
            double dice = rand.nextDouble();
            if (dice < 0.3) {
                loadFactor = 0.1 + rand.nextDouble() * 0.2; // 30%概率: 空闲
            } else if (dice < 0.8) {
                loadFactor = 0.5 + rand.nextDouble() * 0.4; // 50%概率: 繁忙
            } else {
                loadFactor = 0.92 + rand.nextDouble() * 0.08; // 20%概率: 拥堵
            }

            // 云端资源丰富，保持较低负载
            if (dev.getName().toLowerCase().contains("cloud")) {
                loadFactor = 0.05;
            }

            currentCpuLoad.put(dev.getId(), totalMips * loadFactor);
            currentRamLoad.put(dev.getId(), (int)(dev.getHost().getRam() * loadFactor));
        }
        // --- [修改结束] ---

        // 2. 构建任务队列 (DAG 依赖排序)
        Set<String> placedModules = new HashSet<>();

        // 先处理所有请求中已经固定的模块 (如 Client)
        for (PlacementRequest req : requests) {
            Application app = applicationInfo.get(req.getApplicationId());
            for (Map.Entry<String, Integer> entry : req.getPlacedMicroservices().entrySet()) {
                String uniqueName = app.getAppId() + "_" + entry.getKey();
                placedModules.add(uniqueName);

                // 预固定模块也要占用资源
                AppModule mod = app.getModuleByName(entry.getKey());
                if(mod != null) updateSimulatedLoad(entry.getValue(), mod);
            }
        }

        // 循环查找依赖满足的模块加入队列
        boolean progress = true;
        while (progress) {
            progress = false;
            for (PlacementRequest req : requests) {
                Application app = applicationInfo.get(req.getApplicationId());
                for (AppModule mod : app.getModules()) {
                    String uniqueName = app.getAppId() + "_" + mod.getName();
                    if (placedModules.contains(uniqueName)) continue; // 已处理

                    // 检查上游依赖
                    boolean dependenciesMet = true;
                    for (AppEdge edge : app.getEdges()) {
                        // 如果有边指向当前模块 (Tuple.UP)，则源模块必须已放置
                        if (edge.getDestination().equals(mod.getName()) && edge.getDirection() == Tuple.UP) {
                            String sourceUnique = app.getAppId() + "_" + edge.getSource();
                            if (!placedModules.contains(sourceUnique)) {
                                dependenciesMet = false;
                                break;
                            }
                        }
                    }

                    if (dependenciesMet) {
                        placementQueue.add(new QueuedModule(mod.getName(), app.getAppId(), mod));
                        placedModules.add(uniqueName);
                        progress = true;
                    }
                }
            }
        }
        System.out.println("Episode Reset. Queue size: " + placementQueue.size());
    }

    // --- [核心逻辑] 执行动作 (增加惩罚) ---
    private ActionResult executeAction(int actionNodeIndex) {
        if (currentModuleIndex >= placementQueue.size())
            return new ActionResult(null, 0, true);

        // Padding 检查 (选了不存在的节点)
        if (actionNodeIndex >= deployableNodes.size()) {
            return new ActionResult(buildStateRepresentation("Invalid Action (Padding)"), -100.0, false);
        }

        QueuedModule curr = placementQueue.get(currentModuleIndex);
        FogDevice node = deployableNodes.get(actionNodeIndex);

        // 资源检查
        double currentMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
        int currentRam = currentRamLoad.getOrDefault(node.getId(), 0);

        boolean enough = (node.getHost().getTotalMips() - currentMips >= curr.moduleObj.getMips()) &&
                (node.getHost().getRam() - currentRam >= curr.moduleObj.getRam());

        double reward = 0.0;
        String desc;

        if (enough) {
            // 放置成功
            updateSimulatedLoad(node.getId(), curr.moduleObj);
            currentPlacementMap.put(curr.getKey(), node.getId());

            // 奖励设计:
            // 1. 基础奖励 +10
            // 2. 负载均衡奖励: 剩余资源越多，奖励越高 (0~5分)
            double loadBalanceBonus = (1.0 - (currentMips / node.getHost().getTotalMips())) * 5.0;
            reward = 10.0 + loadBalanceBonus;
            desc = "Placed " + curr.moduleName + " (" + curr.appId + ") on " + node.getName();
        } else {
            // 放置失败 (资源不足)
            reward = -100.0; // [修改] 重罚！逼迫 Agent 避开满载节点
            desc = "Failed to place " + curr.moduleName + " on " + node.getName() + " (Resource Full)";
            // 失败策略：当前模块被跳过 (不回退，继续处理下一个)
        }

        currentModuleIndex++;
        boolean done = (currentModuleIndex >= placementQueue.size());

        // 完成奖励：如果顺利跑完且大部分成功，可以给额外奖励 (这里简化处理)
        if (done && reward > 0) reward += 50.0;

        return new ActionResult(buildStateRepresentation(desc), reward, done);
    }

    private void updateSimulatedLoad(int nodeId, AppModule mod) {
        if(mod == null) return;
        currentCpuLoad.put(nodeId, currentCpuLoad.getOrDefault(nodeId, 0.0) + mod.getMips());
        currentRamLoad.put(nodeId, currentRamLoad.getOrDefault(nodeId, 0) + mod.getRam());
    }

    private StateRepresentation buildStateRepresentation() { return buildStateRepresentation(""); }
    private StateRepresentation buildStateRepresentation(String desc) {
        List<Double> state = new ArrayList<>();
        List<Boolean> mask = new ArrayList<>();

        // 1. 节点特征 (Padding 到 MAX_NODES)
        for (int i = 0; i < MAX_NODES; i++) {
            if (i < deployableNodes.size()) {
                FogDevice dev = deployableNodes.get(i);
                double total = dev.getHost().getTotalMips();
                double used = currentCpuLoad.getOrDefault(dev.getId(), 0.0);

                // Feature 1: CPU Usage Ratio (0.0 - 1.0)
                state.add((total - used) / total);
                // Feature 2: Valid Node Flag
                state.add(1.0);
                mask.add(true);
            } else {
                state.add(0.0); state.add(0.0);
                mask.add(false);
            }
        }

        // 2. 当前待部署模块特征
        if (currentModuleIndex < placementQueue.size()) {
            QueuedModule qm = placementQueue.get(currentModuleIndex);
            // 归一化处理 (假设最大MIPS需求 5000, 最大RAM 4096)
            state.add(qm.moduleObj.getMips() / 5000.0);
            state.add(qm.moduleObj.getRam() / 4096.0);
        } else {
            state.add(0.0); state.add(0.0);
        }
        return new StateRepresentation(state, mask, desc);
    }

    // --- 生成最终输出 (含服务发现广播) ---
    private PlacementLogicOutput generateFinalOutput() {
        Map<Integer, Map<Application, List<ModuleLaunchConfig>>> perDevice = new HashMap<>();
        Map<Integer, List<Pair<String, Integer>>> serviceDiscoveryInfo = new HashMap<>();
        List<Pair<String, Integer>> globalServiceList = new ArrayList<>();

        // 将简单的 <String, Integer> 映射转换为 iFogSim 的复杂结构
        for (Map.Entry<String, Integer> entry : currentPlacementMap.entrySet()) {
            int nodeId = entry.getValue();
            // key 是 "AppId_ModuleName"
            String[] parts = entry.getKey().split("_", 2);
            String appId = parts[0];
            String moduleName = parts[1];

            Application app = applicationInfo.get(appId);
            AppModule module = app.getModuleByName(moduleName);

            // 填充 perDevice
            perDevice.putIfAbsent(nodeId, new HashMap<>());
            perDevice.get(nodeId).putIfAbsent(app, new ArrayList<>());
            perDevice.get(nodeId).get(app).add(new ModuleLaunchConfig(module, 1)); // 1 instance

            // 收集服务位置 (用于广播)
            globalServiceList.add(new Pair<>(moduleName, nodeId));
        }

        // 全局广播：告诉网络中所有设备，所有微服务在哪里
        // 解决 "Service Discovery Info Missing" 错误
        for(FogDevice dev : this.fogDevices) {
            serviceDiscoveryInfo.put(dev.getId(), new ArrayList<>(globalServiceList));
        }

        return new PlacementLogicOutput(perDevice, serviceDiscoveryInfo, new HashMap<>());
    }

    // --- HTTP Server 实现 ---
    private void startRestApiServerOnce() {
        if (serverRunning) return;
        try {
            server = HttpServer.create(new InetSocketAddress(API_PORT), 0);
            server.createContext("/reset", ex -> {
                if ("POST".equals(ex.getRequestMethod())) {
                    resetInternalState(placementRequests);
                    sendResponse(ex, gson.toJson(buildStateRepresentation()));
                }
            });
            server.createContext("/step", ex -> {
                if ("POST".equals(ex.getRequestMethod())) {
                    Map<String, Double> body = gson.fromJson(new InputStreamReader(ex.getRequestBody()), new TypeToken<Map<String, Double>>(){}.getType());
                    ActionResult res = executeAction(body.get("action").intValue());
                    sendResponse(ex, gson.toJson(res));
                }
            });
            server.createContext("/get_final_reward", ex -> sendResponse(ex, gson.toJson(new FinalResult(0.0))));
            server.createContext("/stop", ex -> {
                sendResponse(ex, "{\"status\":\"stopped\"}");
                server.stop(0);
                synchronized(RLPlacementLogic.this) { RLPlacementLogic.this.notifyAll(); }
            });
            server.setExecutor(Executors.newCachedThreadPool());
            server.start();
            serverRunning = true;
            System.out.println("API Server started on port " + API_PORT);
        } catch (IOException e) { e.printStackTrace(); }
    }

    private void sendResponse(HttpExchange ex, String resp) throws IOException {
        byte[] bytes = resp.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json");
        ex.sendResponseHeaders(200, bytes.length);
        ex.getResponseBody().write(bytes);
        ex.getResponseBody().close();
    }

    @Override public void updateResources(Map<Integer, Map<String, Double>> r) {}
    @Override public void postProcessing() {}
}