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

    // 训练时请手动改为 false
    // ==========================================================
    public static boolean IS_EVAL_MODE = true;

    // [新增] 用于控制实验一致性的静态变量
    private static final long BASE_SEED = 99999;
    private static int resetCounter = 0;

    // [新增] 用于存储物理仿真结果 (供 Python 读取)
    public static volatile double finalEnergy = -1.0;
    public static volatile double finalMakespan = -1.0;
    public static volatile boolean simulationFinished = false;

    // [新增] 用于控制主线程退出的锁
    public static final Object shutdownLock = new Object();
    private static final int MAX_NODES = 100;
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
        List<Integer> nodeIds;
        StateRepresentation(List<Double> s, List<Boolean> m, String d, List<Integer> ids) {
            stateVector=s; actionMask=m; description=d; nodeIds=ids;
        }
    }

    static class ActionResult {
        StateRepresentation nextStateRepresentation;
        double immediateReward;
        boolean done;
        ActionResult(StateRepresentation s, double r, boolean d) { nextStateRepresentation=s; immediateReward=r; done=d; }
    }
    //新增
    static class SimResult {
        double energy;
        double makespan;
        String status;
        SimResult(double e, double m, String s) { energy=e; makespan=m; status=s; }
    }
    static class FinalResult { double finalReward; FinalResult(double r) { finalReward=r; } }

    public RLPlacementLogic(int fonId) {}

//    @Override
//    public PlacementLogicOutput run(List<FogDevice> fogDevices, Map<String, Application> applicationInfo,
//                                    Map<Integer, Map<String, Double>> resourceAvailability, List<PlacementRequest> pr) {
//
//        // [修复] 绕过 Controller，获取全量 49 个节点
//        List<FogDevice> allDevices = new ArrayList<>();
//        for (Object entity : CloudSim.getEntityList()) {
//            if (entity instanceof FogDevice) allDevices.add((FogDevice) entity);
//        }
//        this.fogDevices = allDevices;
//
//        this.applicationInfo = applicationInfo;
//        this.placementRequests = pr;
//
//        this.fogDeviceMap = new HashMap<>();
//        for (FogDevice d : this.fogDevices) fogDeviceMap.put(d.getId(), d);
//
//        this.deployableNodes = new ArrayList<>();
//        for (FogDevice dev : this.fogDevices) {
//            if (dev.getLevel() <= 2) deployableNodes.add(dev);
//        }
//        this.deployableNodes.sort(Comparator.comparingInt(FogDevice::getId));
//
//        System.out.println("\n=== RL Logic Initialized (FINAL FIXED) ===");
//        System.out.println("Total FogDevices: " + this.fogDevices.size());
//        System.out.println("Deployable Nodes: " + deployableNodes.size());
//        System.out.println("Waiting for Python Agent...");
//
//        System.out.println("\n=== Node Mapping (Action Index -> Node ID) ===");
//        for (int i = 0; i < deployableNodes.size(); i++) {
//            FogDevice dev = deployableNodes.get(i);
//            String type = "EDGE";
//            if (dev.getLevel() == 0) type = "CLOUD";
//            else if (dev.getLevel() == 1) type = "GATEWAY";
//
//            // 使用 %d 来打印整数 (RAM)，这是我们当前最需要确认的信息
//            System.out.printf("Action %d -> ID %d (%s) | MIPS: %d | RAM: %d\n",
//                    i,
//                    dev.getId(),
//                    type,
//                    (int) dev.getHost().getTotalMips(),
//                    dev.getHost().getRam());
//        }
//        System.out.println("==============================================\n");
//
//        startRestApiServerOnce();
//
//        synchronized(this) {
//            try { this.wait(); } catch (InterruptedException e) { e.printStackTrace(); }
//        }
//        return generateFinalOutput();
//    }
    // [RLPlacementLogic.java]

    // ... (前面的成员变量保持不变) ...

    @Override
    public PlacementLogicOutput run(List<FogDevice> fogDevices, Map<String, Application> applicationInfo,
                                    Map<Integer, Map<String, Double>> resourceAvailability, List<PlacementRequest> pr) {

        finalEnergy = -1.0;
        finalMakespan = -1.0;
        simulationFinished = false;

        // 1. 获取全量节点
        List<FogDevice> allDevices = new ArrayList<>();
        for (Object entity : CloudSim.getEntityList()) {
            if (entity instanceof FogDevice) allDevices.add((FogDevice) entity);
        }
        this.fogDevices = allDevices;
        this.applicationInfo = applicationInfo;
        this.placementRequests = pr;

        this.fogDeviceMap = new HashMap<>();
        for (FogDevice d : this.fogDevices) fogDeviceMap.put(d.getId(), d);

        // 保证 Action 0-48 永远是训练时的那些节点，新节点排在 Action 49+
        // ================================================================
        this.deployableNodes = getOrderedDeployableNodes(this.fogDevices);

        System.out.println("\n=== RL Logic Initialized (Topology Locked for Generalization) ===");
        System.out.println("Total FogDevices: " + this.fogDevices.size());
        System.out.println("Deployable Nodes: " + deployableNodes.size());

        // 打印映射表，方便你调试时确认 ID >= 50 的是新节点
        System.out.println("\n=== Node Mapping Verification ===");
        for (int i = 0; i < deployableNodes.size(); i++) {
            FogDevice dev = deployableNodes.get(i);
            String tag = (i >= 49) ? " [NEW]" : ""; // 假设训练时有49个节点
            if (i < 5 || i >= 45) { // 只打印头尾，避免刷屏
                System.out.printf("Action %d -> ID %d (%s)%s\n", i, dev.getId(), dev.getName(), tag);
            }
        }
        System.out.println("=================================\n");

        startRestApiServerOnce();

        synchronized(this) {
            try { this.wait(); } catch (InterruptedException e) { e.printStackTrace(); }
        }
        return generateFinalOutput();
    }

    public static void onSimulationComplete(double energy, double makespan) {
        finalEnergy = energy;
        finalMakespan = makespan;
        simulationFinished = true;
        System.out.println(">>> [RLPlacementLogic] Simulation Finished Signal Received. Energy: " + energy + " Time: " + makespan);
    }
    // --- [新增辅助方法 1] ---
    private List<FogDevice> getOrderedDeployableNodes(List<FogDevice> allDevices) {
        List<FogDevice> orderedList = new ArrayList<>();
        Map<String, FogDevice> nameMap = new HashMap<>();

        for (FogDevice dev : allDevices) {
            if (dev.getLevel() <= 2) nameMap.put(dev.getName(), dev);
        }

        // 1. 先填满训练时的“老坑位” (Cloud + 4 Gateways + 11 Edges/Gateway)
        addIfPresent(orderedList, nameMap, "cloud"); // Action 0
        int trainGateways = 4;
        int trainNodesPerGateway = 11; // ！！！必须写死为训练时的数字 (11)！！！

        for (int i = 0; i < trainGateways; i++) {
            addIfPresent(orderedList, nameMap, "gateway-" + i);
            for (int j = 0; j < trainNodesPerGateway; j++) {
                addIfPresent(orderedList, nameMap, "edge-node-" + i + "-" + j);
            }
        }

        // 2. 再把多出来的“新节点”追加到后面
        List<FogDevice> newNodes = new ArrayList<>();
        for (FogDevice dev : nameMap.values()) {
            if (!orderedList.contains(dev)) {
                newNodes.add(dev);
            }
        }
        // 按 ID 排序，保证确定性
        newNodes.sort(Comparator.comparingInt(FogDevice::getId));
        orderedList.addAll(newNodes);

        return orderedList;
    }

    // --- [新增辅助方法 2] ---
    private void addIfPresent(List<FogDevice> list, Map<String, FogDevice> map, String name) {
        if (map.containsKey(name)) {
            list.add(map.get(name));
        }
    }

    // [修改] resetInternalState 方法
    private void resetInternalState(List<PlacementRequest> requests) {
        // 1. 重置核心数据结构
        this.placementQueue = new LinkedList<>();
        this.currentPlacementMap = new HashMap<>();
        this.currentCpuLoad = new HashMap<>();
        this.currentRamLoad = new HashMap<>();
        this.currentModuleIndex = 0;

//        // [核心修改] 使用确定性随机种子
//        // 这样每次实验(Run)的第一轮、第二轮生成的负载完全一致
//        long currentSeed = BASE_SEED + resetCounter;
//        Random rand = new Random(currentSeed);
//        System.out.println("DEBUG: Resetting Environment with Deterministic Seed: " + currentSeed);
//
//        // 增加计数器，确保同一场实验内的下一轮 Episode 会有变化（避免死循环），
//        // 但重启 Java 后会重置，从而保证 Baseline 和 Ours 面对的是同一组序列。
//        resetCounter++;
//
//        // 2. 打乱请求 (使用相同的 Random 对象，保证打乱顺序一致)
//        List<PlacementRequest> shuffledRequests = new ArrayList<>(requests);
//        Collections.shuffle(shuffledRequests, rand);
//
//        // 3. 生成背景负载
//        // maxBackgroundLoad 在 10% 到 40% 之间波动 (由种子决定)
//        double maxBackgroundLoad = 0.1 + rand.nextDouble() * 0.2;
//
//        System.out.printf("DEBUG: Environment Difficulty (MaxLoad) = %.2f%%\n", maxBackgroundLoad * 100);
//
//        // 初始化节点负载
//        for (FogDevice dev : deployableNodes) {
//            double totalMips = dev.getHost().getTotalMips();
//            double loadFactor = 0.0;
//
//            if (dev.getName().toLowerCase().contains("cloud")) {
//                loadFactor = 0.01;
//            } else if (dev.getName().toLowerCase().contains("gateway")) {
//                loadFactor = 0.1 + rand.nextDouble() * 0.2;
//            } else {
//                // Edge 节点：确定性随机负载
//                loadFactor = rand.nextDouble() * maxBackgroundLoad;
//            }
//
//            currentCpuLoad.put(dev.getId(), totalMips * loadFactor);
//            currentRamLoad.put(dev.getId(), (int)(dev.getHost().getRam() * loadFactor));
//        }
        Random rand;
        // [关键] 根据模式选择随机源
        if (IS_EVAL_MODE) {
            long currentSeed = BASE_SEED + resetCounter;
            rand = new Random(currentSeed);
            System.out.println("DEBUG: [Eval Mode] Reset with Fixed Seed: " + currentSeed);
            resetCounter++;
        } else {
            rand = new Random(); // 训练模式用真随机
            // System.out.println("DEBUG: [Train Mode] Reset with Random Seed");
        }

        // 使用同一个 rand 对象进行 shuffle 和 负载生成
        List<PlacementRequest> shuffledRequests = new ArrayList<>(requests);
        Collections.shuffle(shuffledRequests, rand);

        double maxBackgroundLoad = 0.1 + rand.nextDouble() * 0.2;
        System.out.printf("DEBUG: Environment Difficulty (MaxLoad) = %.2f%%\n", maxBackgroundLoad * 100);

        for (FogDevice dev : deployableNodes) {
            double totalMips = dev.getHost().getTotalMips();
            double loadFactor = 0.0;
            if (dev.getName().toLowerCase().contains("cloud")) loadFactor = 0.01;
            else if (dev.getName().toLowerCase().contains("gateway")) loadFactor = 0.1 + rand.nextDouble() * 0.2;
            else loadFactor = rand.nextDouble() * maxBackgroundLoad;

            currentCpuLoad.put(dev.getId(), totalMips * loadFactor);
            currentRamLoad.put(dev.getId(), (int)(dev.getHost().getRam() * loadFactor));
        }
        // 4. 初始化预部署组件 (Client / Sensor)
        Set<String> placedModules = new HashSet<>();
        for (PlacementRequest req : shuffledRequests) {
            Application app = applicationInfo.get(req.getApplicationId());
            if (app == null) continue;

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

        // 5. 构建任务队列
        boolean progress = true;
        while (progress) {
            progress = false;
            for (PlacementRequest req : shuffledRequests) {
                Application app = applicationInfo.get(req.getApplicationId());
                if (app == null) continue;

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
        // 1. 边界检查
        if (currentModuleIndex >= placementQueue.size()) {
            return new ActionResult(null, 0, true);
        }

        // 如果 Agent 选了非法的动作索引 (比如超出范围)，给巨额惩罚
        if (actionNodeIndex >= deployableNodes.size()) {
            return new ActionResult(buildStateRepresentation("Invalid Action", false), -100.0, false);
        }

        QueuedModule curr = placementQueue.get(currentModuleIndex);
        FogDevice node = deployableNodes.get(actionNodeIndex);

        // 2. 检查资源是否真的足够 (这是物理硬约束，Mask 应该已经挡住了，这里是双重保险)
        double currentMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
        double totalMips = node.getHost().getTotalMips();
        boolean enoughCpu = (totalMips - currentMips) >= curr.moduleObj.getMips();

        int currentRam = currentRamLoad.getOrDefault(node.getId(), 0);
        int totalRam = node.getHost().getRam();
        boolean enoughRam = (totalRam - currentRam) >= curr.moduleObj.getRam();

        double reward = 0.0;
        String desc;

        if (enoughCpu && enoughRam) {
            // === 部署成功，执行状态更新 ===
            updateSimulatedLoad(node.getId(), curr.moduleObj);
            currentPlacementMap.put(curr.getKey(), node.getId());

            // =============================================================
            // 目标：在 "局部性(低延迟)" 和 "负载均衡(防拥堵)" 之间博弈
            // =============================================================

            // --- 1. 局部性奖励 (Locality Reward) ---
            // 逻辑：找到当前微服务的上游(Predecessor)，看它在哪
            double transmissionReward = 0.0;
            Application app = applicationInfo.get(curr.appId);

            for (AppEdge edge : app.getEdges()) {
                // 找到指向当前模块的边 (Tuple.UP)
                if (edge.getDestination().equals(curr.moduleName) && edge.getDirection() == Tuple.UP) {
                    String sourceKey = curr.appId + "_" + edge.getSource();
                    // 处理特殊的源名称
                    if (edge.getSource().equals("client")) sourceKey = curr.appId + "_client";
                    else if (edge.getSource().startsWith("s-")) sourceKey = edge.getSource(); // sensor

                    if (currentPlacementMap.containsKey(sourceKey)) {
                        int sourceId = currentPlacementMap.get(sourceKey);

                        if (sourceId == node.getId()) {
                            // [完美] 同一节点，无网络开销
                            transmissionReward += 20.0;
                        } else {
                            FogDevice sourceNode = fogDeviceMap.get(sourceId);
                            // 检查是否在同一个网关下 (Parent 相同)
                            if (sourceNode != null && sourceNode.getParentId() == node.getParentId() && sourceNode.getParentId() != -1) {
                                // [不错] 同邻居，延迟较低
                                transmissionReward += 10.0;
                            } else {
                                // [差] 跨网关或跨层级，产生高延迟
                                transmissionReward -= 10.0;
                            }
                        }
                    }
                }
            }

            // --- 2. 负载均衡惩罚 (Load Balancing Penalty) ---
            // 逻辑：如果节点变得太拥挤 (>70%)，开始给予非线性惩罚
            // 迫使 RL 在节点快满时，主动放弃"局部性"，去寻找新的空闲节点
            double newUtilization = (currentMips + curr.moduleObj.getMips()) / totalMips;
            double loadPenalty = 0.0;

            if (newUtilization > 0.9) {
                loadPenalty = -60.0; // 极度危险，接近满载，重罚
            } else if (newUtilization > 0.7) {
                // 指数级增长的惩罚: 0.7->0, 0.8->-10, 0.9->-40
                loadPenalty = Math.pow((newUtilization - 0.7) * 20, 2) * -1.0;
            }

            // --- 3. 基础生存分 (Base Reward) ---
            double baseReward = 50.0;
            boolean isCloud = node.getName().toLowerCase().contains("cloud");

            if (isCloud) {
                // Cloud 只有低保分，除非所有 Edge 都挤爆了(-60 penalty)，否则 RL 不会选 Cloud
                baseReward = 5.0;
            }

            // --- 4. 总分计算 ---
            // 理想情况 (Edge空闲+同节点): 50 + 0 + 20 = 70
            // 拥堵情况 (Edge满载+同节点): 50 - 60 + 20 = 10 (不如去空闲的远端)
            // 兜底情况 (Cloud): 5 + 0 - 10 = -5 (比失败强)
            reward = baseReward + transmissionReward + loadPenalty;

            desc = String.format("Placed %s on %s | Base:%.0f Link:%+.1f LoadPen:%.1f | R: %.2f",
                    curr.moduleName, node.getName(), baseReward, transmissionReward, loadPenalty, reward);

            // 打印日志 (可选，如果不希望刷屏可以注释掉)
//            System.out.println(desc);

        } else {
            // === 部署失败 ===
            // 即使 Mask 挡住了大部分，但如果是 Cloud 也没资源了(极其罕见)，或者并发冲突，这里做兜底
            // 给一个比 Cloud 略低的惩罚，但不要太低，以免训练震荡
            reward = -50.0;
            desc = "Failed (Resource)";
        }

        // 推进到下一个微服务
        currentModuleIndex++;
        boolean done = (currentModuleIndex >= placementQueue.size());

        // 全局完成奖励 (可选：给一个大大的赞)
        if (done && reward > 0) reward += 10.0;

        // 生成下一个状态
        // 注意：这里调用的是 buildStateRepresentation
        return new ActionResult(buildStateRepresentation(desc, true), reward, done);
    }

    private void updateSimulatedLoad(int nodeId, AppModule mod) {
        if(mod == null) return;
        currentCpuLoad.put(nodeId, currentCpuLoad.getOrDefault(nodeId, 0.0) + mod.getMips());
        currentRamLoad.put(nodeId, currentRamLoad.getOrDefault(nodeId, 0) + mod.getRam());
    }

//    // [增强版] 生成环境快照 (Prompt) - 补充 RAM 和 链路信息
//    private String generateEnvironmentDescription(QueuedModule curr) {
//        StringBuilder sb = new StringBuilder();
//
//        // 1. 任务基本需求
//        sb.append(String.format("Current Task: %s (App %s). Requirements: %.0f MIPS, %d RAM.\n",
//                curr.moduleName, curr.appId, curr.moduleObj.getMips(), curr.moduleObj.getRam()));
//
//        // 2. 链路上下文：告诉 LLM 前置服务在哪里
//        String predecessorLoc = "Unknown";
//        Application app = applicationInfo.get(curr.appId);
//        if (app != null) {
//            for (AppEdge edge : app.getEdges()) {
//                // 找到指向当前模块的边 (Upstream)
//                if (edge.getDestination().equals(curr.moduleName) && edge.getDirection() == Tuple.UP) {
//                    String sourceName = edge.getSource();
//                    String sourceKey = curr.appId + "_" + sourceName;
//
//                    // 处理特殊的源名称
//                    if (sourceName.equals("client")) sourceKey = curr.appId + "_client";
//                    else if (sourceName.startsWith("s-")) sourceKey = sourceName; // sensor
//
//                    if (currentPlacementMap.containsKey(sourceKey)) {
//                        int prevNodeId = currentPlacementMap.get(sourceKey);
//                        predecessorLoc = String.format("Node %d", prevNodeId);
//                    } else {
//                        predecessorLoc = "Not Placed Yet / Sensor";
//                    }
//                    break;
//                }
//            }
//        }
//        sb.append(String.format("Data Source (Predecessor) is located at: %s.\n", predecessorLoc));
//
//        // 3. 节点状态列表
//        sb.append("Nodes Status (Top 15 relevant):\n");
//        for (FogDevice node : deployableNodes) {
//
//            // CPU 信息
//            double currentMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
//            double totalMips = node.getHost().getTotalMips();
//            double freeMips = totalMips - currentMips;
//
//            // [新增] RAM 信息
//            int totalRam = node.getHost().getRam();
//            int usedRam = currentRamLoad.getOrDefault(node.getId(), 0);
//            int freeRam = totalRam - usedRam;
//
//            // 过滤掉几乎不可用的节点，减少 Prompt 长度
//            if (freeMips < 100 && !node.getName().contains("cloud")) continue;
//
//            String type = node.getName().contains("cloud") ? "Cloud" : "Edge";
//
//            // [优化] 输出格式包含 RAM
//            sb.append(String.format("- ID %d (%s): Free CPU %.0f/%.0f, Free RAM %d/%d.\n",
//                    node.getId(), type, freeMips, totalMips, freeRam, totalRam));
//        }
//        return sb.toString();
//    }
// [增强版] 生成环境快照 (Prompt) - 补充 RAM 和 拓扑链路信息
private String generateEnvironmentDescription(QueuedModule curr) {
    StringBuilder sb = new StringBuilder();

    // 1. 任务基本需求
    sb.append(String.format("Current Task: %s (App %s). Requirements: %.0f MIPS, %d RAM.\n",
            curr.moduleName, curr.appId, curr.moduleObj.getMips(), curr.moduleObj.getRam()));

    // 2. 链路上下文：寻找前置节点
    int prevNodeId = -1;
    String predecessorLoc = "Unknown";
    Application app = applicationInfo.get(curr.appId);
    if (app != null) {
        for (AppEdge edge : app.getEdges()) {
            if (edge.getDestination().equals(curr.moduleName) && edge.getDirection() == Tuple.UP) {
                String sourceName = edge.getSource();
                String sourceKey = curr.appId + "_" + sourceName;

                if (sourceName.equals("client")) sourceKey = curr.appId + "_client";
                else if (sourceName.startsWith("s-")) sourceKey = sourceName;

                if (currentPlacementMap.containsKey(sourceKey)) {
                    prevNodeId = currentPlacementMap.get(sourceKey);
                    predecessorLoc = String.format("Node %d", prevNodeId);
                } else {
                    predecessorLoc = "Not Placed Yet / Sensor";
                }
                break;
            }
        }
    }
    sb.append(String.format("Data Source (Predecessor) is located at: %s.\n", predecessorLoc));

    // 3. 节点状态列表 (必须包含 Link 标签)
    sb.append("Nodes Status:\n");
    for (FogDevice node : deployableNodes) {

        double currentMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
        double totalMips = node.getHost().getTotalMips();
        double freeMips = totalMips - currentMips;

        int totalRam = node.getHost().getRam();
        int usedRam = currentRamLoad.getOrDefault(node.getId(), 0);
        int freeRam = totalRam - usedRam;

        // 过滤掉几乎不可用的节点 (除了 Cloud)
        if (freeMips < 100 && !node.getName().contains("cloud")) continue;

        String type = node.getName().contains("cloud") ? "Cloud" : "Edge";

        // === [核心逻辑] 计算相对距离标签 ===
        String linkStatus = "Remote";

        if (node.getName().toLowerCase().contains("cloud")) {
            linkStatus = "Cloud";
        } else if (prevNodeId != -1) {
            if (node.getId() == prevNodeId) {
                linkStatus = "Local"; // 完美：同节点
            } else {
                FogDevice prevNode = fogDeviceMap.get(prevNodeId);
                // 检查是否在同一个网关下 (ParentID 相同)
                if (prevNode != null && node.getParentId() == prevNode.getParentId() && node.getParentId() != -1) {
                    linkStatus = "Neighbor"; // 优秀：同网关
                }
            }
        } else if (predecessorLoc.contains("Sensor") || predecessorLoc.contains("client")) {
            // 简化处理：对于 Sensor/Client 直连的 Gateway 下的节点，通常是 Neighbor
            // 这里暂且保守标记，具体可根据实际情况优化
        }

        // [输出] 增加 Link: XXX
        sb.append(String.format("- ID %d (%s): Free CPU %.0f/%.0f, Free RAM %d/%d, Link: %s.\n",
                node.getId(), type, freeMips, totalMips, freeRam, totalRam, linkStatus));
    }
    return sb.toString();
}


private StateRepresentation buildStateRepresentation(String logDesc, boolean isPreDecision) {
    List<Double> state = new ArrayList<>();
    List<Boolean> mask = new ArrayList<>();

    // 1. 获取当前任务
    QueuedModule currentTask = null;
    double reqMips = 0;
    String predecessorKey = null;

    if (currentModuleIndex < placementQueue.size()) {
        currentTask = placementQueue.get(currentModuleIndex);
        reqMips = currentTask.moduleObj.getMips();

        // 寻找前置节点 Key
        Application app = applicationInfo.get(currentTask.appId);
        for (AppEdge edge : app.getEdges()) {
            if (edge.getDestination().equals(currentTask.moduleName) && edge.getDirection() == Tuple.UP) {
                String sourceName = edge.getSource();
                if (sourceName.equals("client")) predecessorKey = currentTask.appId + "_client";
                else if (sourceName.startsWith("s-")) predecessorKey = sourceName;
                else predecessorKey = currentTask.appId + "_" + sourceName;
                break;
            }
        }
    }

    int prevNodeId = -1;
    if (predecessorKey != null && currentPlacementMap.containsKey(predecessorKey)) {
        prevNodeId = currentPlacementMap.get(predecessorKey);
    }

    // 2. 遍历节点生成 3 维特征
    for (int i = 0; i < MAX_NODES; i++) {
        if (i < deployableNodes.size()) {
            FogDevice dev = deployableNodes.get(i);
            double totalMips = dev.getHost().getTotalMips();
            double usedMips = currentCpuLoad.getOrDefault(dev.getId(), 0.0);
            int totalRam = dev.getHost().getRam();
            int usedRam = currentRamLoad.getOrDefault(dev.getId(), 0);

            // --- 特征 1: 负载压力 (Load Pressure) ---
            double loadPressure = usedMips / totalMips;

            // --- 特征 2: 链路代价 (Link Cost) ---
            double linkCost = 0.5;
            if (prevNodeId != -1) {
                if (dev.getId() == prevNodeId) linkCost = 0.0;
                else if (fogDeviceMap.get(dev.getId()).getParentId() == fogDeviceMap.get(prevNodeId).getParentId()) linkCost = 0.2;
                else linkCost = 0.5;
            }
            if (dev.getName().contains("cloud")) linkCost = 1.0;

            // --- 特征 3: 资源余量评分 (Margin Ratio) ---
            // 剩余资源是需求的几倍？归一化到 [0, 1]
            double freeMips = totalMips - usedMips;
            double marginRatio = 0.0;
            if (reqMips > 0 && reqMips <= freeMips) {
                marginRatio = Math.min((freeMips / reqMips) / 5.0, 1.0);
            } else if (reqMips > 0) {
                marginRatio = -1.0; // 表示资源不足
            }

            state.add(loadPressure); // Dim 1
            state.add(linkCost);     // Dim 2
            state.add(marginRatio);  // Dim 3

            // --- Mask (硬约束) ---
            boolean canDeploy = false;
            if (dev.getName().contains("cloud")) {
                // Cloud 可以有更大余量，但也要检查
                canDeploy = (freeMips >= reqMips * 0.8);
            } else {
                // Edge 严格检查
                canDeploy = (freeMips >= reqMips * 1.0);
            }
            mask.add(canDeploy);

        } else {
            // Padding
            state.add(1.0); state.add(1.0); state.add(0.0);
            mask.add(false);
        }
    }

    // 任务特征
    if (currentTask != null) {
        state.add(reqMips / 5000.0);
        state.add(0.0);
    } else {
        state.add(0.0); state.add(0.0);
    }

    String finalDesc = (isPreDecision && currentTask != null) ? generateEnvironmentDescription(currentTask) : "";
    // [新增] 收集当前 deployableNodes 的 ID 顺序
    List<Integer> currentIds = new ArrayList<>();
    for (FogDevice dev : deployableNodes) {
        currentIds.add(dev.getId());
    }
    // 如果有 Padding，补 -1
    while (currentIds.size() < MAX_NODES) {
        currentIds.add(-1);
    }
    // 修改 return 语句，传入 currentIds
    return new StateRepresentation(state, mask, finalDesc, currentIds);
}

    private PlacementLogicOutput generateFinalOutput() {
        Map<Integer, Map<Application, List<ModuleLaunchConfig>>> perDevice = new HashMap<>();
        Map<Integer, List<Pair<String, Integer>>> serviceDiscoveryInfo = new HashMap<>();
        List<Pair<String, Integer>> globalServiceList = new ArrayList<>();

        for (Map.Entry<String, Integer> entry : currentPlacementMap.entrySet()) {
            int nodeId = entry.getValue();
            String[] parts = entry.getKey().split("_", 2);
            if (parts.length < 2 || parts[1].equals("sensor") || parts[1].startsWith("s-")) {
                continue;
            }
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
        // 在generateFinalOutput方法中添加分析

       // 计算负载均衡指标
        Map<Integer, Integer> nodeLoadCount = new HashMap<>();
        for (Map.Entry<String, Integer> entry : currentPlacementMap.entrySet()) {
            if (entry.getKey().contains("sensor") || entry.getKey().contains("client"))
                continue;
            int nodeId = entry.getValue();
            nodeLoadCount.put(nodeId, nodeLoadCount.getOrDefault(nodeId, 0) + 1);
        }

//        System.out.println("\n=== 负载均衡分析 ===");
//        System.out.println("节点ID | 微服务数量 | 建议阈值");
//        for (Map.Entry<Integer, Integer> entry : nodeLoadCount.entrySet()) {
//            FogDevice dev = fogDeviceMap.get(entry.getKey());
//            if (dev != null) {
//                double realUsedMips = currentCpuLoad.getOrDefault(entry.getKey(), 0.0);
//                double estimatedUtil = realUsedMips / dev.getHost().getTotalMips();
//                String warning = estimatedUtil > 0.8 ? "⚠过载" : "✓正常";
//                System.out.printf("%6d | %10d | %s (预计利用率: %.1f%%)\n",
//                        entry.getKey(), entry.getValue(), warning, estimatedUtil * 100);
//            }
//        }
        System.out.println("\n=== 负载均衡分析 (含背景流量) ===");
        // 增加显示 Total MIPS，让你看清背景负载
        System.out.printf("%-6s | %-8s | %-20s | %-8s | %-6s%n", "NodeID", "SvcCount", "Used/Total MIPS", "Util%", "Status");
        System.out.println("---------------------------------------------------------------");

        // [关键修复] 遍历所有可部署节点 (deployableNodes)，而不是只遍历已部署的 map
        // 这样你才能看到那些因为背景负载太高而被 RL 放弃的节点！
        for (FogDevice dev : deployableNodes) {
            int nodeId = dev.getId();

            // 获取 RL 放置的服务数量 (如果没有就是 0)
            int serviceCount = nodeLoadCount.getOrDefault(nodeId, 0);

            // 获取 实际总负载 (背景 + RL)
            // currentCpuLoad 在 resetInternalState 时已经包含了随机背景负载，所以这里的数据是真实的！
            double realUsedMips = currentCpuLoad.getOrDefault(nodeId, 0.0);
            double totalMips = dev.getHost().getTotalMips();
            double util = realUsedMips / totalMips;

            // 状态标记
            String status = "✓OK";
            if (util > 0.95) status = "🔥FULL";
            else if (util > 0.8) status = "⚠High";

            // 为了版面整洁，只打印利用率 > 1% 的节点 (过滤掉纯空的 Cloud 等)
            // 这样你就能看到：虽然 SvcCount=0，但 Util% 可能是 80% (背景流量)
            if (util > 0.01 || serviceCount > 0) {
                System.out.printf("%-6d | %-8d | %8.0f / %-8.0f | %5.1f%%   | %s%n",
                        nodeId, serviceCount, realUsedMips, totalMips, util * 100.0, status);
            }
        }
        System.out.println("---------------------------------------------------------------");
        // [修改结束] -----------------------------------------------------------
        // 计算共置指标
        Map<String, Set<Integer>> appDeploymentNodes = new HashMap<>();
        for (Map.Entry<String, Integer> entry : currentPlacementMap.entrySet()) {
            String[] parts = entry.getKey().split("_", 2);
            if (parts.length < 2) continue;
            String appId = parts[0];
            if (!appId.startsWith("A")) continue;

            appDeploymentNodes.putIfAbsent(appId, new HashSet<>());
            appDeploymentNodes.get(appId).add(entry.getValue());
        }

        System.out.println("\n=== 应用共置分析 ===");
        System.out.println("应用ID | 使用节点数 | 建议");
        for (Map.Entry<String, Set<Integer>> entry : appDeploymentNodes.entrySet()) {
            String suggestion = entry.getValue().size() <= 2 ? "良好" : "可优化";
            System.out.printf("%6s | %10d | %s\n", entry.getKey(), entry.getValue().size(), suggestion);
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
            // [新增] /start_simulation 接口 (Eval模式用)
            server.createContext("/start_simulation", ex -> {
                String resp = "{\"status\":\"sim_started\"}";
                ex.sendResponseHeaders(200, resp.length());
                ex.getResponseBody().write(resp.getBytes());
                ex.getResponseBody().close();
                // 唤醒主线程去跑 CloudSim
                synchronized(RLPlacementLogic.this) { RLPlacementLogic.this.notifyAll(); }
            });

            // [新增] /get_results 接口 (Eval模式用)
            server.createContext("/get_results", ex -> {
                SimResult res;
                if (simulationFinished) {
                    res = new SimResult(finalEnergy, finalMakespan, "finished");
                } else {
                    res = new SimResult(-1.0, -1.0, "running");
                }
                byte[] bytes = gson.toJson(res).getBytes(StandardCharsets.UTF_8);
                ex.sendResponseHeaders(200, bytes.length);
                ex.getResponseBody().write(bytes);
                ex.getResponseBody().close();
            });

            // [新增] /shutdown 接口 (Eval模式用)
            server.createContext("/shutdown", ex -> {
                String resp = "{\"status\":\"shutdown\"}";
                ex.sendResponseHeaders(200, resp.length());
                ex.getResponseBody().write(resp.getBytes());
                ex.getResponseBody().close();
                server.stop(0);
                System.out.println(">>> [Eval Mode] Shutdown signal received. Exiting.");
                System.exit(0); // [关键] 强制杀掉进程，防止 CloudSim 线程卡死
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