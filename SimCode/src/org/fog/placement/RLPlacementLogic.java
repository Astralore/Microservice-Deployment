package org.fog.placement;

import org.apache.commons.math3.util.Pair;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.fog.application.AppEdge;
import org.fog.application.AppModule;
import org.fog.application.Application;
import org.fog.entities.FogDevice;
import org.fog.entities.PlacementRequest;
import org.fog.entities.Tuple;
import org.fog.utils.ModuleLaunchConfig;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.Executors;

/**
 * 强化学习部署逻辑类 (Modified for Latency, Energy & Link Affinity)
 */
public class RLPlacementLogic implements MicroservicePlacementLogic {

    // --- [配置] 必须与 Python config.py 一致 ---
    private static final int MAX_NODES = 50;
    private static final int API_PORT = 4567;

    // --- 成员变量 ---
    private List<FogDevice> fogDevices;
    private List<FogDevice> deployableNodes;
    private Map<Integer, FogDevice> fogDeviceMap;
    private Map<String, Application> applicationInfo;
    private List<PlacementRequest> placementRequests;

    // 内部状态
    private LinkedList<QueuedModule> placementQueue;
    private Map<String, Integer> currentPlacementMap;
    private Map<Integer, Double> currentCpuLoad;
    private Map<Integer, Integer> currentRamLoad;
    private int currentModuleIndex = 0;

    private HttpServer server;
    private Gson gson = new Gson();
    private static volatile boolean serverRunning = false;

    // --- 辅助类 ---
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

    public RLPlacementLogic(int fonId) {}

    @Override
    public PlacementLogicOutput run(List<FogDevice> fogDevices, Map<String, Application> applicationInfo,
                                    Map<Integer, Map<String, Double>> resourceAvailability, List<PlacementRequest> pr) {

        // --- [核心修复] 绕过 Controller 传入的可能有问题的列表，直接从 CloudSim 获取全量 ---
        System.out.println("DEBUG: Input list size from Controller: " + fogDevices.size());

        List<FogDevice> allDevices = new ArrayList<>();
        for (Object entity : CloudSim.getEntityList()) {
            if (entity instanceof FogDevice) {
                allDevices.add((FogDevice) entity);
            }
        }
        System.out.println("DEBUG: Global CloudSim entity list size: " + allDevices.size());
        this.fogDevices = allDevices;
        // -----------------------------------------------------------------

        this.applicationInfo = applicationInfo;
        this.placementRequests = pr;

        this.fogDeviceMap = new HashMap<>();
        for (FogDevice d : this.fogDevices) fogDeviceMap.put(d.getId(), d);

        // 筛选可部署节点 (Level <= 2)
        this.deployableNodes = new ArrayList<>();
        for (FogDevice dev : this.fogDevices) {
            if (dev.getLevel() <= 2) deployableNodes.add(dev);
        }
        this.deployableNodes.sort(Comparator.comparingInt(FogDevice::getId));

        // 调试打印
        System.out.println("\n=== RL Logic Initialized (FIXED) ===");
        System.out.println("Total FogDevices (Global): " + this.fogDevices.size());
        System.out.println("Deployable Nodes: " + deployableNodes.size());
        if (!deployableNodes.isEmpty()) {
            System.out.println(" - First: [" + deployableNodes.get(0).getId() + "] " + deployableNodes.get(0).getName());
            System.out.println(" - Last:  [" + deployableNodes.get(deployableNodes.size()-1).getId() + "] " + deployableNodes.get(deployableNodes.size()-1).getName());
        }
        System.out.println("==============================\n");

        System.out.println("Waiting for Python Agent to connect...");
        startRestApiServerOnce();

        synchronized(this) {
            try { this.wait(); } catch (InterruptedException e) { e.printStackTrace(); }
        }

        System.out.println("RL Training finished. Generating final placement for iFogSim...");
        return generateFinalOutput();
    }

    private void resetInternalState(List<PlacementRequest> requests) {
        this.placementQueue = new LinkedList<>();
        this.currentPlacementMap = new HashMap<>();
        this.currentCpuLoad = new HashMap<>();
        this.currentRamLoad = new HashMap<>();
        this.currentModuleIndex = 0;

        Random rand = new Random();
        for (FogDevice dev : deployableNodes) {
            double totalMips = dev.getHost().getTotalMips();
            double loadFactor = 0.0;
            double dice = rand.nextDouble();
            if (dice < 0.3) loadFactor = 0.1 + rand.nextDouble() * 0.2;
            else if (dice < 0.8) loadFactor = 0.5 + rand.nextDouble() * 0.4;
            else loadFactor = 0.92 + rand.nextDouble() * 0.08;

            if (dev.getName().toLowerCase().contains("cloud")) loadFactor = 0.05;

            currentCpuLoad.put(dev.getId(), totalMips * loadFactor);
            currentRamLoad.put(dev.getId(), (int)(dev.getHost().getRam() * loadFactor));
        }

        Set<String> placedModules = new HashSet<>();
        for (PlacementRequest req : requests) {
            Application app = applicationInfo.get(req.getApplicationId());

            // [新增] 记录 Sensor/Client 的位置，供后续链路计算使用
            // 假设 Sensor 名字通常包含 "sensor" 或者就是 req 中指定的 gateway 设备
            // 这里我们把 req 中的 gatewayId (即 sensor 所在的边缘节点) 记录为该 App 的源头
            String clientKey = app.getAppId() + "_client";
            String sensorKey = "s-" + app.getAppId(); // Sensor 名字通常是 s-A0
            // 注意：MicroservicePlacement.java 里 sensor 也是放在 req.getGatewayDeviceId()
            currentPlacementMap.put(clientKey, req.getGatewayDeviceId());
            currentPlacementMap.put(sensorKey, req.getGatewayDeviceId());
            // 为了更稳健，记录一下 "sensor" 通用名
            currentPlacementMap.put(app.getAppId() + "_sensor", req.getGatewayDeviceId());

            for (Map.Entry<String, Integer> entry : req.getPlacedMicroservices().entrySet()) {
                String uniqueName = app.getAppId() + "_" + entry.getKey();
                placedModules.add(uniqueName);
                AppModule mod = app.getModuleByName(entry.getKey());
                if(mod != null) updateSimulatedLoad(entry.getValue(), mod);
            }
        }

        boolean progress = true;
        while (progress) {
            progress = false;
            for (PlacementRequest req : requests) {
                Application app = applicationInfo.get(req.getApplicationId());
                for (AppModule mod : app.getModules()) {
                    String uniqueName = app.getAppId() + "_" + mod.getName();
                    if (placedModules.contains(uniqueName)) continue;

                    boolean dependenciesMet = true;
                    for (AppEdge edge : app.getEdges()) {
                        if (edge.getDestination().equals(mod.getName()) && edge.getDirection() == Tuple.UP) {
                            String sourceUnique = app.getAppId() + "_" + edge.getSource();
                            // 注意：Sensor/Client 已经在上面预填充进 map 了
                            if (!currentPlacementMap.containsKey(sourceUnique) && !placedModules.contains(sourceUnique)) {
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

    private ActionResult executeAction(int actionNodeIndex) {
        // 1. 越界与队列检查
        if (currentModuleIndex >= placementQueue.size())
            return new ActionResult(null, 0, true);

        // Padding 处理：如果选了不存在的节点，给予重罚
        if (actionNodeIndex >= deployableNodes.size()) {
            return new ActionResult(buildStateRepresentation("Invalid Action"), -100.0, false);
        }

        QueuedModule curr = placementQueue.get(currentModuleIndex);
        FogDevice node = deployableNodes.get(actionNodeIndex);

        // 2. 获取当前负载数据
        double currentMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
        double totalMips = node.getHost().getTotalMips();

        // 3. 资源预判 (CPU & RAM)
        // 只有当剩余资源足够容纳当前微服务时，才算部署成功
        boolean enoughCpu = (totalMips - currentMips) >= curr.moduleObj.getMips();
        boolean enoughRam = (node.getHost().getRam() - currentRamLoad.getOrDefault(node.getId(), 0)) >= curr.moduleObj.getRam();

        double reward = 0.0;
        String desc;

        if (enoughCpu && enoughRam) {
            // 更新模拟环境的负载状态
            updateSimulatedLoad(node.getId(), curr.moduleObj);
            // 记录决策结果 (用于链路亲和性计算)
            currentPlacementMap.put(curr.getKey(), node.getId());

            // A. 基础生存分 (Base Survival Reward)
            // 只要能活下来（部署成功），就给 50 分。这保证了正向行为是正分。
            double baseReward = 50.0;

            // B. 节点类型判断 (Cloud vs Edge)
            boolean isCloud = node.getName().toLowerCase().contains("cloud");

            // C. 时延惩罚 (Latency Penalty)
            // 云端：距离远，扣 20 分。
            // 边缘：距离近，扣 0 分 (相当于相对奖励)。
            double latencyPenalty = 0.0;
            if (isCloud) {
                latencyPenalty = 20.0;
            } else {
                latencyPenalty = 0.0;
            }
            // D. 能耗惩罚 (Energy Penalty) - 归一化到 0-10 分
            // Low Spec: Idle 50W, Busy 80W
            // High Spec: Idle 180W, Busy 250W
            double idlePwr = 50.0; // 默认低配
            double busyPwr = 80.0;
            if (totalMips > 3500) { // 高配节点
                busyPwr = 250.0;
                idlePwr = 180.0;
            } else if (totalMips > 2500) { // 中配节点
                busyPwr = 120.0;
                idlePwr = 85.0;
            }

            // 计算该任务占用的功耗份额
            double cpuUsageFraction = curr.moduleObj.getMips() / totalMips;
            double estimatedPowerCost = (busyPwr - idlePwr) * cpuUsageFraction + idlePwr * 0.1; // 加上一点待机基数
            // 映射：假设最大能耗代价约为 50W-100W，我们将其缩放到 0-10 分
            // 250W 的节点跑满可能扣 10 分，50W 的节点跑满扣 2 分
            double energyPenalty = (estimatedPowerCost / 100.0) * 5.0;
            // 限制最大惩罚不超过 10
            if(energyPenalty > 10.0) energyPenalty = 10.0;

            // E. 链路亲和性惩罚 (Link Affinity) - 归一化到 0-15 分
            // 避免流量乒乓：上游在哪，我最好也在哪
            double transmissionPenalty = 0.0;
            Application app = applicationInfo.get(curr.appId);

            for (AppEdge edge : app.getEdges()) {
                if (edge.getDestination().equals(curr.moduleName) && edge.getDirection() == Tuple.UP) {
                    String sourceName = edge.getSource();
                    String sourceKey = curr.appId + "_" + sourceName;

                    // 处理 Client/Sensor 的命名差异
                    if (sourceName.equals("client")) sourceKey = curr.appId + "_client";
                    else if (sourceName.startsWith("s-")) sourceKey = sourceName;
                    else if (sourceName.equals("sensor")) sourceKey = curr.appId + "_sensor";

                    if (currentPlacementMap.containsKey(sourceKey)) {
                        int sourceId = currentPlacementMap.get(sourceKey);
                        FogDevice sourceNode = fogDeviceMap.get(sourceId);

                        if (sourceNode != null) {
                            if (sourceId == node.getId()) {
                                transmissionPenalty += 0.0; // 同节点，完美 (0)
                            } else if (sourceNode.getParentId() == node.getParentId() && sourceNode.getParentId() != -1) {
                                transmissionPenalty += 5.0; // 同网关邻居，尚可 (-5)
                            } else {
                                transmissionPenalty += 15.0; // 跨网关/跨云边，差 (-15)
                            }
                        }
                    }
                }
            }

            // F. 负载均衡奖励 (0-5 分)
            double newUtilization = (currentMips + curr.moduleObj.getMips()) / totalMips;
            double lbBonus = (1.0 - newUtilization) * 5.0;

            // --- 最终公式汇总 ---
            // 理想情况 (边缘本地): 50 + 5 - 2 - 0 - 0 = 53
            // 兜底情况 (云端):     50 + 5 - 5 - 20 - 15 = 15
            // 失败情况:           -100
            reward = baseReward + lbBonus - energyPenalty - latencyPenalty - transmissionPenalty;

            desc = String.format("Placed %s on %s | Type:%s | Lat:-%.1f Pwr:-%.1f Link:-%.1f | R: %.2f",
                    curr.moduleName, node.getName(), (isCloud?"CLOUD":"EDGE"),
                    latencyPenalty, energyPenalty, transmissionPenalty, reward);

            // 打印决策详情，用于调试
            System.out.println(desc);

        } else {
            // --- 部署失败 (资源不足) ---
            reward = -100.0;
            desc = "Failed to place " + curr.moduleName + " on " + node.getName() + " (Full/Incompatible)";
        }

        currentModuleIndex++;
        boolean done = (currentModuleIndex >= placementQueue.size());

        if (done && reward > 0) reward += 5.0;

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

        for (int i = 0; i < MAX_NODES; i++) {
            if (i < deployableNodes.size()) {
                FogDevice dev = deployableNodes.get(i);
                double total = dev.getHost().getTotalMips();
                double used = currentCpuLoad.getOrDefault(dev.getId(), 0.0);
                // 1. CPU 剩余率 (Availability)
                state.add((total - used) / total);

                // 2. 节点绝对能力 (Capacity) - 归一化
                // 假设最大 MIPS 约 5000, 最大 RAM 约 8192
                state.add(total / 5000.0);
                state.add(dev.getHost().getRam() / 8192.0);

                // 3. 节点层级 (Location)
                state.add(dev.getLevel() / 2.0);

                mask.add(true);
            } else {
                state.add(0.0); state.add(0.0); state.add(0.0); state.add(0.0);
                mask.add(false);
            }
        }

        if (currentModuleIndex < placementQueue.size()) {
            QueuedModule qm = placementQueue.get(currentModuleIndex);
            state.add(qm.moduleObj.getMips() / 5000.0);
            state.add(qm.moduleObj.getRam() / 4096.0);
        } else {
            state.add(0.0); state.add(0.0);
        }
        return new StateRepresentation(state, mask, desc);
    }

    private PlacementLogicOutput generateFinalOutput() {
        Map<Integer, Map<Application, List<ModuleLaunchConfig>>> perDevice = new HashMap<>();
        Map<Integer, List<Pair<String, Integer>>> serviceDiscoveryInfo = new HashMap<>();
        List<Pair<String, Integer>> globalServiceList = new ArrayList<>();

        for (Map.Entry<String, Integer> entry : currentPlacementMap.entrySet()) {
            int nodeId = entry.getValue();
            String[] parts = entry.getKey().split("_", 2);
            // 过滤掉 sensor/client 这种辅助记录
            if (parts.length < 2 || parts[1].equals("sensor") || parts[1].equals("client") || parts[1].startsWith("s-")) continue;

            String appId = parts[0];
            String moduleName = parts[1];
            Application app = applicationInfo.get(appId);
            if (app == null) continue;

            AppModule module = app.getModuleByName(moduleName);
            if (module == null) continue;

            perDevice.putIfAbsent(nodeId, new HashMap<>());
            perDevice.get(nodeId).putIfAbsent(app, new ArrayList<>());
            perDevice.get(nodeId).get(app).add(new ModuleLaunchConfig(module, 1));

            globalServiceList.add(new Pair<>(moduleName, nodeId));
        }

        for(FogDevice dev : this.fogDevices) {
            serviceDiscoveryInfo.put(dev.getId(), new ArrayList<>(globalServiceList));
        }

        return new PlacementLogicOutput(perDevice, serviceDiscoveryInfo, new HashMap<>());
    }

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