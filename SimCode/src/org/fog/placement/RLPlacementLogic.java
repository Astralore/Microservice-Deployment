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

public class RLPlacementLogic implements MicroservicePlacementLogic {

    private static final int MAX_NODES = 50;
    private static final int API_PORT = 4567;

    private List<FogDevice> fogDevices;
    private List<FogDevice> deployableNodes;
    private Map<Integer, FogDevice> fogDeviceMap;
    private Map<String, Application> applicationInfo;
    private List<PlacementRequest> placementRequests;

    private LinkedList<QueuedModule> placementQueue;
    private Map<String, Integer> currentPlacementMap;
    private Map<Integer, Double> currentCpuLoad;
    private Map<Integer, Integer> currentRamLoad;
    private int currentModuleIndex = 0;

    private HttpServer server;
    private Gson gson = new Gson();
    private static volatile boolean serverRunning = false;

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

        // [修复] 绕过 Controller，获取全量 49 个节点
        List<FogDevice> allDevices = new ArrayList<>();
        for (Object entity : CloudSim.getEntityList()) {
            if (entity instanceof FogDevice) allDevices.add((FogDevice) entity);
        }
        this.fogDevices = allDevices;

        this.applicationInfo = applicationInfo;
        this.placementRequests = pr;

        this.fogDeviceMap = new HashMap<>();
        for (FogDevice d : this.fogDevices) fogDeviceMap.put(d.getId(), d);

        this.deployableNodes = new ArrayList<>();
        for (FogDevice dev : this.fogDevices) {
            if (dev.getLevel() <= 2) deployableNodes.add(dev);
        }
        this.deployableNodes.sort(Comparator.comparingInt(FogDevice::getId));

        System.out.println("\n=== RL Logic Initialized (FINAL FIXED) ===");
        System.out.println("Total FogDevices: " + this.fogDevices.size());
        System.out.println("Deployable Nodes: " + deployableNodes.size());
        System.out.println("Waiting for Python Agent...");

        System.out.println("\n=== Node Mapping (Action Index -> Node ID) ===");
        for (int i = 0; i < deployableNodes.size(); i++) {
            FogDevice dev = deployableNodes.get(i);
            String type = "EDGE";
            if (dev.getLevel() == 0) type = "CLOUD";
            else if (dev.getLevel() == 1) type = "GATEWAY";

            // 使用 %d 来打印整数 (RAM)，这是我们当前最需要确认的信息
            System.out.printf("Action %d -> ID %d (%s) | MIPS: %d | RAM: %d\n",
                    i,
                    dev.getId(),
                    type,
                    (int) dev.getHost().getTotalMips(),
                    dev.getHost().getRam());
        }
        System.out.println("==============================================\n");

        startRestApiServerOnce();

        synchronized(this) {
            try { this.wait(); } catch (InterruptedException e) { e.printStackTrace(); }
        }
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

            if (dev.getName().toLowerCase().contains("cloud")) {
                loadFactor = 0.01;
            } else if (dev.getName().toLowerCase().contains("gateway")) {
                loadFactor = 0.1 + rand.nextDouble() * 0.2;
            } else {
                // 边缘节点 0% - 25% 随机负载
                loadFactor = rand.nextDouble() * 0.30;
            }

            currentCpuLoad.put(dev.getId(), totalMips * loadFactor);
            currentRamLoad.put(dev.getId(), (int)(dev.getHost().getRam() * loadFactor));
        }
//        for (FogDevice dev : deployableNodes) {
//            // 初始负载设为 0 (完全空闲)
//            currentCpuLoad.put(dev.getId(), 0.0);
//            currentRamLoad.put(dev.getId(), 0);
//        }
        Set<String> placedModules = new HashSet<>();
        for (PlacementRequest req : requests) {
            Application app = applicationInfo.get(req.getApplicationId());

            String clientKey = app.getAppId() + "_client";
            String sensorKey = "s-" + app.getAppId();
            currentPlacementMap.put(clientKey, req.getGatewayDeviceId());
            currentPlacementMap.put(sensorKey, req.getGatewayDeviceId());
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
    }

    private ActionResult executeAction(int actionNodeIndex) {
        if (currentModuleIndex >= placementQueue.size()) return new ActionResult(null, 0, true);
        if (actionNodeIndex >= deployableNodes.size()) return new ActionResult(buildStateRepresentation("Invalid", false), -100.0, false);

        QueuedModule curr = placementQueue.get(currentModuleIndex);
        FogDevice node = deployableNodes.get(actionNodeIndex);

        double currentMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
        double totalMips = node.getHost().getTotalMips();
        boolean enoughCpu = (totalMips - currentMips) >= curr.moduleObj.getMips();
        boolean enoughRam = (node.getHost().getRam() - currentRamLoad.getOrDefault(node.getId(), 0)) >= curr.moduleObj.getRam();

        double reward = 0.0;
        String desc;
        if (enoughCpu && enoughRam) {
            updateSimulatedLoad(node.getId(), curr.moduleObj);
            currentPlacementMap.put(curr.getKey(), node.getId());

            double baseReward = 100.0;
            boolean isCloud = node.getName().toLowerCase().contains("cloud");

            // 时延惩罚
            double latencyPenalty = 0.0;
            if (isCloud) {
                latencyPenalty = 95.0;
            } else {
                latencyPenalty = 0.0;
            }

            double idlePwr = 50.0, busyPwr = 80.0;
            if (totalMips > 3500) { busyPwr = 250.0; idlePwr = 180.0; }
            else if (totalMips > 2500) { busyPwr = 120.0; idlePwr = 85.0; }

            double cpuUsageFraction = curr.moduleObj.getMips() / totalMips;
            double estimatedPowerCost = (busyPwr - idlePwr) * cpuUsageFraction + idlePwr * 0.1;
            double energyPenalty = (estimatedPowerCost / 100.0) * 5.0;
            if(energyPenalty > 10.0) energyPenalty = 10.0;

            double transmissionPenalty = 0.0;
            Application app = applicationInfo.get(curr.appId);
            for (AppEdge edge : app.getEdges()) {
                if (edge.getDestination().equals(curr.moduleName) && edge.getDirection() == Tuple.UP) {
                    String sourceKey = curr.appId + "_" + edge.getSource();
                    if (edge.getSource().equals("client")) sourceKey = curr.appId + "_client";
                    else if (edge.getSource().startsWith("s-")) sourceKey = edge.getSource();
                    else if (edge.getSource().equals("sensor")) sourceKey = curr.appId + "_sensor";

                    if (currentPlacementMap.containsKey(sourceKey)) {
                        int sourceId = currentPlacementMap.get(sourceKey);
                        FogDevice sourceNode = fogDeviceMap.get(sourceId);
                        if (sourceNode != null) {
                            if (sourceId == node.getId()) transmissionPenalty += 0.0;
                            else if (sourceNode.getParentId() == node.getParentId() && sourceNode.getParentId() != -1) transmissionPenalty += 5.0;
                            else transmissionPenalty += 15.0;
                        }
                    }
                }
            }

            double newUtilization = (currentMips + curr.moduleObj.getMips()) / totalMips;
            double lbBonus = (1.0 - newUtilization) * 5.0;

            reward = baseReward + lbBonus - energyPenalty - latencyPenalty - transmissionPenalty;
            desc = String.format("Placed %s on %s | Type:%s | Lat:-%.1f Pwr:-%.1f Link:-%.1f | R: %.2f",
                    curr.moduleName, node.getName(), (isCloud?"CLOUD":"EDGE"),
                    latencyPenalty, energyPenalty, transmissionPenalty, reward);
            System.out.println(desc);

        } else {
            reward = -60.0;
            desc = "Failed (Resource)";
        }

        currentModuleIndex++;
        boolean done = (currentModuleIndex >= placementQueue.size());
        if (done && reward > 0) reward += 10.0;

        return new ActionResult(buildStateRepresentation(desc, false), reward, done);
    }

    private void updateSimulatedLoad(int nodeId, AppModule mod) {
        if(mod == null) return;
        currentCpuLoad.put(nodeId, currentCpuLoad.getOrDefault(nodeId, 0.0) + mod.getMips());
        currentRamLoad.put(nodeId, currentRamLoad.getOrDefault(nodeId, 0) + mod.getRam());
    }

    // [新增] 生成环境快照 (Prompt)
//    private String generateEnvironmentDescription(QueuedModule curr) {
//        StringBuilder sb = new StringBuilder();
//        sb.append(String.format("Current Task: %s (App %s). Requirements: %.0f MIPS, %d RAM.\n",
//                curr.moduleName, curr.appId, curr.moduleObj.getMips(), curr.moduleObj.getRam()));
//        sb.append("Nodes Status (Top 15 relevant):\n");
//
//        int count = 0;
//        for (FogDevice node : deployableNodes) {
//            // 简单筛选：只展示 Cloud 和有一定空闲的 Edge，防止Prompt太长
//            if (count > 15) break;
//
//            double currentMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
//            double totalMips = node.getHost().getTotalMips();
//            double freeMips = totalMips - currentMips;
//
//            // 跳过那些已经彻底满载的低配节点，减少噪声
//            if (freeMips < 100 && !node.getName().contains("cloud")) continue;
//
//            String type = node.getName().contains("cloud") ? "Cloud" : "Edge";
//            sb.append(String.format("- ID %d (%s): Total %.0f MIPS, Free %.0f.\n",
//                    node.getId(), type, totalMips, freeMips));
//            count++;
//        }
//        return sb.toString();
//    }
    // [增强版] 生成环境快照 (Prompt) - 补充 RAM 和 链路信息
    private String generateEnvironmentDescription(QueuedModule curr) {
        StringBuilder sb = new StringBuilder();

        // 1. 任务基本需求
        sb.append(String.format("Current Task: %s (App %s). Requirements: %.0f MIPS, %d RAM.\n",
                curr.moduleName, curr.appId, curr.moduleObj.getMips(), curr.moduleObj.getRam()));

        // 2. 链路上下文：告诉 LLM 前置服务在哪里
        String predecessorLoc = "Unknown";
        Application app = applicationInfo.get(curr.appId);
        if (app != null) {
            for (AppEdge edge : app.getEdges()) {
                // 找到指向当前模块的边 (Upstream)
                if (edge.getDestination().equals(curr.moduleName) && edge.getDirection() == Tuple.UP) {
                    String sourceName = edge.getSource();
                    String sourceKey = curr.appId + "_" + sourceName;

                    // 处理特殊的源名称
                    if (sourceName.equals("client")) sourceKey = curr.appId + "_client";
                    else if (sourceName.startsWith("s-")) sourceKey = sourceName; // sensor

                    if (currentPlacementMap.containsKey(sourceKey)) {
                        int prevNodeId = currentPlacementMap.get(sourceKey);
                        predecessorLoc = String.format("Node %d", prevNodeId);
                    } else {
                        predecessorLoc = "Not Placed Yet / Sensor";
                    }
                    break;
                }
            }
        }
        sb.append(String.format("Data Source (Predecessor) is located at: %s.\n", predecessorLoc));

        // 3. 节点状态列表
        sb.append("Nodes Status (Top 15 relevant):\n");
        int count = 0;
        for (FogDevice node : deployableNodes) {
            if (count > 15) break;

            // CPU 信息
            double currentMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
            double totalMips = node.getHost().getTotalMips();
            double freeMips = totalMips - currentMips;

            // [新增] RAM 信息
            int totalRam = node.getHost().getRam();
            int usedRam = currentRamLoad.getOrDefault(node.getId(), 0);
            int freeRam = totalRam - usedRam;

            // 过滤掉几乎不可用的节点，减少 Prompt 长度
            if (freeMips < 100 && !node.getName().contains("cloud")) continue;

            String type = node.getName().contains("cloud") ? "Cloud" : "Edge";

            // [优化] 输出格式包含 RAM
            sb.append(String.format("- ID %d (%s): Free CPU %.0f/%.0f, Free RAM %d/%d.\n",
                    node.getId(), type, freeMips, totalMips, freeRam, totalRam));
            count++;
        }
        return sb.toString();
    }

//    // [修改] 增加 isPreDecision 参数
//    private StateRepresentation buildStateRepresentation(String logDesc, boolean isPreDecision) {
//        List<Double> state = new ArrayList<>();
//        List<Boolean> mask = new ArrayList<>();
//
//        for (int i = 0; i < MAX_NODES; i++) {
//            if (i < deployableNodes.size()) {
//                FogDevice dev = deployableNodes.get(i);
//                double total = dev.getHost().getTotalMips();
//                double used = currentCpuLoad.getOrDefault(dev.getId(), 0.0);
//
//                state.add((total - used) / total);
//                state.add(total / 5000.0);
//                state.add(dev.getHost().getRam() / 8192.0);
//                state.add(dev.getLevel() / 2.0);
//
//                mask.add(true);
//            } else {
//                state.add(0.0); state.add(0.0); state.add(0.0); state.add(0.0);
//                mask.add(false);
//            }
//        }
//
//        if (currentModuleIndex < placementQueue.size()) {
//            QueuedModule qm = placementQueue.get(currentModuleIndex);
//            state.add(qm.moduleObj.getMips() / 5000.0);
//            state.add(qm.moduleObj.getRam() / 4096.0);
//        } else {
//            state.add(0.0); state.add(0.0);
//        }
//
//        // [关键] 决定返回给 Python 的 description
//        String finalDesc = "";
//        if (isPreDecision && currentModuleIndex < placementQueue.size()) {
//            // 决策前：生成环境快照
//            finalDesc = generateEnvironmentDescription(placementQueue.get(currentModuleIndex));
//        } else {
//            // 决策后：返回日志，或者为了下一个 Step 预生成下一个环境描述
//            // 注意：Step 返回的 Info 包含的是 Next State 的描述
//            if (currentModuleIndex < placementQueue.size()) {
//                finalDesc = generateEnvironmentDescription(placementQueue.get(currentModuleIndex));
//            } else {
//                finalDesc = "Episode Finished";
//            }
//        }
//
//        return new StateRepresentation(state, mask, finalDesc);
//    }


    //使用mask的方案
//private StateRepresentation buildStateRepresentation(String logDesc, boolean isPreDecision) {
//    List<Double> state = new ArrayList<>();
//    List<Boolean> mask = new ArrayList<>();
//
//    // 1. 获取当前等待部署的任务（如果有）
//    QueuedModule currentTask = null;
//    if (currentModuleIndex < placementQueue.size()) {
//        currentTask = placementQueue.get(currentModuleIndex);
//    }
//
//    for (int i = 0; i < MAX_NODES; i++) {
//        if (i < deployableNodes.size()) {
//            FogDevice dev = deployableNodes.get(i);
//
//            // --- 获取资源信息 ---
//            double totalMips = dev.getHost().getTotalMips();
//            double usedMips = currentCpuLoad.getOrDefault(dev.getId(), 0.0);
//            double freeMips = totalMips - usedMips;
//
//            int totalRam = dev.getHost().getRam();
//            int usedRam = currentRamLoad.getOrDefault(dev.getId(), 0);
//            int freeRam = totalRam - usedRam;
//
//            // --- 构建状态向量 (State) ---
//            // 使用 Math.min 稍微限制一下归一化数值，防止 Cloud 数值过大主导梯度
//            state.add(Math.min(freeMips / totalMips, 1.0)); // 剩余 CPU 比例
//            state.add(Math.min(totalMips / 5000.0, 5.0));   // 绝对 CPU 能力 (限制上限)
//            state.add(Math.min(totalRam / 8192.0, 2.0));    // RAM 能力
//            state.add(dev.getLevel() / 2.0);                // 层级归一化
//
//            // ---构建掩码 (Mask) ---
//            boolean isDeployable = true;
//
//            if (currentTask != null) {
//                double reqMips = currentTask.moduleObj.getMips();
//                int reqRam = currentTask.moduleObj.getRam();
//                // 核心逻辑：只有 CPU 和 RAM 都足够时，Mask 才为 true
//                // 如果资源不足，Mask 为 false，Agent 就根本无法选择这个节点
//                if (freeMips < reqMips || freeRam < reqRam) {
//                    isDeployable = false;
//                }
//            }
//            // 防止所有节点都满了导致死锁 (虽然 Cloud 很难满)
//            if (dev.getName().toLowerCase().contains("cloud") && freeMips > 100) {
//                isDeployable = true;
//            }
//
//            mask.add(isDeployable);
//
//        } else {
//            // Padding 部分，Mask 必须为 false
//            state.add(0.0); state.add(0.0); state.add(0.0); state.add(0.0);
//            mask.add(false);
//        }
//    }
//
//    // 添加任务本身的特征
//    if (currentModuleIndex < placementQueue.size()) {
//        QueuedModule qm = placementQueue.get(currentModuleIndex);
//        state.add(qm.moduleObj.getMips() / 5000.0);
//        state.add(qm.moduleObj.getRam() / 4096.0);
//    } else {
//        state.add(0.0); state.add(0.0);
//    }
//
//    // 生成 LLM 描述
//    String finalDesc = "";
//    if (isPreDecision && currentModuleIndex < placementQueue.size()) {
//        finalDesc = generateEnvironmentDescription(placementQueue.get(currentModuleIndex));
//    } else {
//        if (currentModuleIndex < placementQueue.size()) {
//            finalDesc = generateEnvironmentDescription(placementQueue.get(currentModuleIndex));
//        } else {
//            finalDesc = "Episode Finished";
//        }
//    }
//
//    return new StateRepresentation(state, mask, finalDesc);
//}



//去掉mask的方案
private StateRepresentation buildStateRepresentation(String logDesc, boolean isPreDecision) {
    List<Double> state = new ArrayList<>();
    List<Boolean> mask = new ArrayList<>();

    // 1. 获取当前等待部署的任务信息
    QueuedModule currentTask = null;
    double reqMips = 0;
    int reqRam = 0;
    if (currentModuleIndex < placementQueue.size()) {
        currentTask = placementQueue.get(currentModuleIndex);
        reqMips = currentTask.moduleObj.getMips();
        reqRam = currentTask.moduleObj.getRam();
    }

    for (int i = 0; i < MAX_NODES; i++) {
        if (i < deployableNodes.size()) {
            FogDevice dev = deployableNodes.get(i);

            // --- 获取资源数据 ---
            double totalMips = dev.getHost().getTotalMips();
            double usedMips = currentCpuLoad.getOrDefault(dev.getId(), 0.0);
            double freeMips = totalMips - usedMips;

            int totalRam = dev.getHost().getRam();
            int usedRam = currentRamLoad.getOrDefault(dev.getId(), 0);
            int freeRam = totalRam - usedRam;
//            // 特征 1: 剩余资源比例 (原有)
//            state.add(Math.min(freeMips / totalMips, 1.0));
//            // 特征 2: CPU 余量 (Margin)
//            // 公式: (剩余 - 需求) / 归一化因子
//            // 逻辑: 正数代表够用，负数代表不够。RL 对正负号非常敏感，一学就会。
//            state.add((freeMips - reqMips) / totalMips);
//            // 特征 3:  RAM 余量 (Margin)
//            state.add((freeRam - reqRam) /(double) totalRam);
//            // 特征 4: 层级 (区分 Cloud/Edge)
//            state.add(dev.getLevel() / 2.0);
//            // --- 撤销 Mask，让 RL 自己判断 ---
//            // 始终为 true。RL 必须学会：如果特征2或特征3是负数，选了就会死(-100)。
//            mask.add(true);

            // 特征 1: 资源是否充足的“红绿灯”信号 (Binary Flag)
            // 如果够用就是 1.0，不够用就是 -1.0。
            // 神经网络学这个比学 0.0002 容易一万倍。
            double cpuSafe = (freeMips >= reqMips) ? 1.0 : -1.0;
            double ramSafe = (freeRam >= reqRam) ? 1.0 : -1.0;
//            // [插入这段 Debug 代码] =======================================
//            if (i == 5 && currentTask != null) { // 只看 ID 为 5 的边缘节点
//                System.out.println("\n>>> [JAVA DEBUG Node 5] <<<");
//                System.out.printf("TotalMips: %.2f, Used: %.2f, Free: %.2f\n", totalMips, usedMips, freeMips);
//                System.out.printf("TaskReq: %.2f\n", reqMips);
//                System.out.printf("Calculated CPU_SAFE: %.1f\n", cpuSafe);
//                System.out.println("===============================\n");
//            }
//            // ==========================================================

            // 为了保留具体还有多少的细粒度信息，我们可以组合一下
            // 格式：[CPU_Safe_Flag, RAM_Safe_Flag, CPU_Ratio, Level]
            state.add(cpuSafe); // 特征 1: CPU 安全标识
            state.add(ramSafe); // 特征 2: RAM 安全标识
            state.add(Math.min(freeMips / totalMips, 1.0)); // 特征 3: 剩余比例 (辅助决策)
            state.add(dev.getLevel() / 2.0); // 特征 4: 层级
            mask.add(true);

        } else {
            // Padding 部分
            state.add(0.0); state.add(0.0); state.add(0.0); state.add(0.0);
            mask.add(false);
        }
    }

    // 任务特征 (保留)
    if (currentTask != null) {
        state.add(reqMips / 5000.0);
        state.add(reqRam / 4096.0);
    } else {
        state.add(0.0); state.add(0.0);
    }

    // 生成 LLM 描述
    String finalDesc = "";
    if (isPreDecision && currentModuleIndex < placementQueue.size()) {
        finalDesc = generateEnvironmentDescription(placementQueue.get(currentModuleIndex));
    } else {
        if (currentModuleIndex < placementQueue.size()) {
            finalDesc = generateEnvironmentDescription(placementQueue.get(currentModuleIndex));
        } else {
            finalDesc = "Episode Finished";
        }
    }

    return new StateRepresentation(state, mask, finalDesc);
}

    private PlacementLogicOutput generateFinalOutput() {
        Map<Integer, Map<Application, List<ModuleLaunchConfig>>> perDevice = new HashMap<>();
        Map<Integer, List<Pair<String, Integer>>> serviceDiscoveryInfo = new HashMap<>();
        List<Pair<String, Integer>> globalServiceList = new ArrayList<>();

        for (Map.Entry<String, Integer> entry : currentPlacementMap.entrySet()) {
            int nodeId = entry.getValue();
            String[] parts = entry.getKey().split("_", 2);
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
        for(FogDevice dev : this.fogDevices) serviceDiscoveryInfo.put(dev.getId(), new ArrayList<>(globalServiceList));
        // [新增] 打印最终部署方案报表 (Human-Readable Report)
        // =========================================================================
        System.out.println("\n\n");
        System.out.println("################################################################");
        System.out.println("#                 FINAL RL DEPLOYMENT REPORT                   #");
        System.out.println("################################################################");
        System.out.printf("%-10s | %-15s | %-10s | %-10s%n", "App ID", "Microservice", "Node ID", "Node Type");
        System.out.println("----------------------------------------------------------------");

        // 对 Key 进行排序 (A0_mService1, A0_mService2...)
        List<String> sortedKeys = new ArrayList<>(currentPlacementMap.keySet());
        Collections.sort(sortedKeys);

        int edgeCount = 0;
        int cloudCount = 0;
        int gatewayCount = 0;

        for (String key : sortedKeys) {
            // 过滤掉 sensor 和 client，我们只关心核心微服务的去向
            if (key.contains("sensor") || key.contains("client") || key.startsWith("s-")) continue;

            int nodeId = currentPlacementMap.get(key);
            String[] parts = key.split("_");
            String appId = parts[0];
            String moduleName = (parts.length > 1) ? parts[1] : key;

            // 判断节点类型 (根据 ID 范围推断，需根据您实际 ID 修改，通常 Cloud=2)
            String nodeType = "EDGE";
            FogDevice device = fogDeviceMap.get(nodeId);

            if (device != null) {
                if (device.getName().toLowerCase().contains("cloud")) {
                    nodeType = "\u001B[31mCLOUD\u001B[0m"; // 红色高亮
                    cloudCount++;
                } else if (device.getName().toLowerCase().contains("gateway")) {
                    nodeType = "\u001B[33mGATEWAY\u001B[0m"; // 黄色高亮
                    gatewayCount++;
                } else {
                    nodeType = "\u001B[32mEDGE\u001B[0m";   // 绿色高亮
                    edgeCount++;
                }
            }

            System.out.printf("%-10s | %-15s | %-10d | %s%n", appId, moduleName, nodeId, nodeType);
        }
        System.out.println("----------------------------------------------------------------");
        System.out.println("Summary Statistics:");
        System.out.println("  - Edge    : " + edgeCount);
        System.out.println("  - Gateway : " + gatewayCount);
        System.out.println("  - Cloud   : " + cloudCount);
        System.out.println("################################################################\n\n");
        // =========================================================================

        return new PlacementLogicOutput(perDevice, serviceDiscoveryInfo, new HashMap<>());
    }

    private void startRestApiServerOnce() {
        if (serverRunning) return;
        try {
            server = HttpServer.create(new InetSocketAddress(API_PORT), 0);
            server.createContext("/reset", ex -> {
                if ("POST".equals(ex.getRequestMethod())) {
                    resetInternalState(placementRequests);
                    // [修改] Reset 返回初始环境描述 (PreDecision = true)
                    byte[] bytes = gson.toJson(buildStateRepresentation("", true)).getBytes(StandardCharsets.UTF_8);
                    ex.sendResponseHeaders(200, bytes.length);
                    ex.getResponseBody().write(bytes);
                    ex.getResponseBody().close();
                }
            });
            server.createContext("/step", ex -> {
                if ("POST".equals(ex.getRequestMethod())) {
                    Map<String, Double> body = gson.fromJson(new InputStreamReader(ex.getRequestBody()), new TypeToken<Map<String, Double>>(){}.getType());
                    ActionResult res = executeAction(body.get("action").intValue());
                    byte[] bytes = gson.toJson(res).getBytes(StandardCharsets.UTF_8);
                    ex.sendResponseHeaders(200, bytes.length);
                    ex.getResponseBody().write(bytes);
                    ex.getResponseBody().close();
                }
            });
            server.createContext("/get_final_reward", ex -> {
                byte[] bytes = gson.toJson(new FinalResult(0.0)).getBytes(StandardCharsets.UTF_8);
                ex.sendResponseHeaders(200, bytes.length);
                ex.getResponseBody().write(bytes);
                ex.getResponseBody().close();
            });
            server.createContext("/stop", ex -> {
                String resp = "{\"status\":\"stopped\"}";
                ex.sendResponseHeaders(200, resp.length());
                ex.getResponseBody().write(resp.getBytes());
                ex.getResponseBody().close();
                server.stop(0);
                synchronized(RLPlacementLogic.this) { RLPlacementLogic.this.notifyAll(); }
            });
            server.setExecutor(Executors.newCachedThreadPool());
            server.start();
            serverRunning = true;
            System.out.println("API Server started on port " + API_PORT);
        } catch (IOException e) { e.printStackTrace(); }
    }

    @Override public void updateResources(Map<Integer, Map<String, Double>> r) {}
    @Override public void postProcessing() {}
}