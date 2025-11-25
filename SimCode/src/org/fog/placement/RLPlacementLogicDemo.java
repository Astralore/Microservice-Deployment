package org.fog.placement;

import com.google.gson.Gson;
import org.fog.application.AppEdge;
import org.fog.application.AppModule;
import org.fog.application.Application;
import org.fog.entities.*;
import org.fog.utils.Logger;
import org.fog.utils.ModuleLaunchConfig;
import org.apache.commons.math3.util.Pair; // 需要 Pair 类
import org.fog.test.perfeval.MicroservicePlacement;

import java.util.*;
import java.util.stream.Collectors;
import java.util.concurrent.CountDownLatch; // 用于 run 方法等待

// 引入 SparkJava
//import static spark.Spark.*;

/**
 * 强化学习部署逻辑实现类.
 * 该类作为 iFogSim 环境与外部 Python RL Agent 交互的桥梁.
 * 它通过 REST API 提供状态信息、执行动作并计算奖励.
 */
public class RLPlacementLogicDemo implements MicroservicePlacementLogic {

    // --- 环境和配置信息 ---
    private List<FogDevice> fogDevices;             // 包含所有 FogDevice (Cloud, Gateway, Edge, Client)
    private List<FogDevice> deployableNodes;        // 仅包含可部署节点 (Cloud, Gateway, Edge Nodes)
    private Map<Integer, FogDevice> fogDeviceMap;   // ID 到 FogDevice 的映射，方便查找
    private Map<String, Application> applicationInfo;
    private List<PlacementRequest> placementRequests; // 初始部署请求
    private Application currentApp;                 // 当前处理的应用 (简化假设：一次处理一个)
    private int fonId;                              // 当前 FON ID (可能未使用)

    // --- 内部状态，模拟逐步部署过程 ---
    private Map<String, Integer> currentPlacementMap; // <ModuleName, NodeId> 存储当前临时的放置决策
    private Map<Integer, Double> currentCpuLoad;      // <NodeId, Current MIPS Load>
    private Map<Integer, Integer> currentRamLoad;      // <NodeId, Current RAM Load (MB)>
    private List<String> modulesToPlaceOrder;         // 按依赖关系确定的模块放置顺序
    private int currentModuleIndex;                   // 指向 modulesToPlaceOrder 中下一个要放置的模块

    // --- REST API 和 JSON 处理 ---
    private static final int API_PORT = 4567;         // API 监听端口
    private Gson gson = new Gson();                   // JSON 处理器
    private static boolean serverRunning = false;     // 标记 API 服务器是否已启动
    private static final Object serverLock = new Object(); // 用于服务器启动同步

    // --- 奖励函数权重 (需要根据实验调优) ---
    private double w_comm = 1.0;    // 通信开销权重
    private double w_bal = 0.5;     // 负载均衡权重
    // private double w_lat = 0.0;  // 端到端延迟权重 (可选)
    // private double w_cost = 0.0; // 资源成本权重 (可选)
    private double penalty_invalid_action = -1.0;   // 无效动作的即时惩罚
    private double penalty_infeasible = -1000.0; // 最终方案不可行的惩罚

    // --- 同步控制 ---
    private CountDownLatch episodeLatch; // 用于 run 方法等待 Agent 完成一轮决策

    // --- 单例模式支持 ---
    private static RLPlacementLogicDemo instance = null;
    private static final Object instanceLock = new Object();

    // 私有构造函数
    public RLPlacementLogicDemo(int fonId) {
        this.fonId = fonId;
        // 延迟启动 API 服务器，在 run 方法中首次调用时启动
    }

    // 获取单例实例的方法
    public static RLPlacementLogicDemo getInstance(int fonId) {
        if (instance == null) {
            synchronized (instanceLock) {
                if (instance == null) {
                    instance = new RLPlacementLogicDemo(fonId);
                }
            }
        }
        // 可以选择是否更新 fonId: instance.fonId = fonId;
        return instance;
    }


//    /**
//     * 初始化并启动 REST API 服务器 (仅执行一次).
//     */
//    private void startRestApiServerOnce() {
//        synchronized (serverLock) {
//            if (!serverRunning) {
//                System.out.println("Starting REST API server on port " + API_PORT + "...");
//                port(API_PORT);
//                // Endpoint: 重置环境
//                // 接收: (可选) JSON body 指定要处理的 Application ID
//                // 返回: JSON StateRepresentation { stateVector: [], actionMask: [] }
//                post("/reset", (request, response) -> {
//                    try {
//                        // TODO: 可以从 request body 获取特定 appId，目前默认处理第一个请求的应用
//                        resetInternalState(this.placementRequests);
//                        StateRepresentation sr = buildStateRepresentation();
//                        response.type("application/json");
//                        System.out.println("/reset called. Initial state sent.");
//                        return gson.toJson(sr);
//                    } catch (Exception e) {
//                        e.printStackTrace();
//                        response.status(500);
//                        return "Error during reset: " + e.getMessage();
//                    }
//                });
//
//                // Endpoint: 执行一步动作
//                // 接收: JSON body {"action": nodeIndex} (nodeIndex 是 deployableNodes 的索引)
//                // 返回: JSON ActionResult { nextStateRepresentation: {...}, immediateReward: R_imm, done: bool }
//                post("/step", (request, response) -> {
//                    try {
//                        Map<String, Double> payload = gson.fromJson(request.body(), Map.class);
//                        if (payload == null || !payload.containsKey("action")) {
//                            response.status(400);
//                            return "Missing 'action' in request body";
//                        }
//                        int actionNodeIndex = payload.get("action").intValue();
//
//                        ActionResult result = executeAction(actionNodeIndex);
//                        response.type("application/json");
//                        //System.out.println("/step called with action index " + actionNodeIndex + ". Done: " + result.done);
//                        return gson.toJson(result);
//                    } catch (Exception e) {
//                        e.printStackTrace();
//                        response.status(500);
//                        return "Error processing step: " + e.getMessage();
//                    }
//                });
//
//                // Endpoint: 获取最终奖励 (当 done=true 后调用)
//                // 接收: 无
//                // 返回: JSON FinalResult { finalReward: R_final }
//                get("/get_final_reward", (request, response) -> {
//                    try {
//                        double finalReward = calculateFinalReward();
//                        response.type("application/json");
//                        System.out.println("/get_final_reward called. Final Reward: " + finalReward);
//                        // 当 Agent 获取最终奖励后，认为一个 Episode 完成，释放 run 方法的等待锁
//                        if (episodeLatch != null) {
//                            episodeLatch.countDown();
//                        }
//                        return gson.toJson(new FinalResult(finalReward));
//                    } catch (Exception e) {
//                        e.printStackTrace();
//                        response.status(500);
//                        return "Error calculating final reward: " + e.getMessage();
//                    }
//                });
//
//                // Endpoint: 停止服务器 (用于仿真结束)
//                get("/stop", (request, response) -> {
//                    System.out.println("/stop called. Stopping server...");
//                    stopServer(); // 调用停止服务器的方法
//                    return "Server stopped";
//                });
//
//                // 处理 API 未找到的情况
//                notFound((req, res) -> {
//                    res.type("application/json");
//                    return "{\"message\":\"API endpoint not found\"}";
//                });
//
//                // 处理内部服务器错误
//                internalServerError((req, res) -> {
//                    res.type("application/json");
//                    return "{\"message\":\"Internal server error\"}";
//                });
//
//                awaitInitialization(); // 等待服务器启动完成
//                serverRunning = true;
//                System.out.println("REST API server started successfully.");
//
//            } else {
//                System.out.println("REST API server already running.");
//            }
//        }
//    }
//
//    /**
//     * 停止 REST API 服务器.
//     */
//    private void stopServer() {
//        synchronized (serverLock) {
//            if (serverRunning) {
//                System.out.println("Stopping REST API server...");
//                spark.Spark.stop(); // 停止 Spark 服务器
//                awaitStop(); // 等待服务器完全停止
//                serverRunning = false;
//                System.out.println("REST API server stopped.");
//            }
//        }
//    }


    @Override
    public PlacementLogicOutput run(List<FogDevice> fogDevices, Map<String, Application> applicationInfo, Map<Integer, Map<String, Double>> resourceAvailability, List<PlacementRequest> pr) {
        this.fogDevices = fogDevices;
        this.applicationInfo = applicationInfo;
        this.placementRequests = pr;

        // 构建 fogDeviceMap
        this.fogDeviceMap = new HashMap<>();
        for (FogDevice device : fogDevices) {
            this.fogDeviceMap.put(device.getId(), device);
        }

        // 筛选可部署节点 (Level 0: Cloud, Level 1: Gateway, Level 2: Edge Node)
        this.deployableNodes = fogDevices.stream()
                .filter(d -> d.getLevel() >= 0 && d.getLevel() <= 2)
                .sorted(Comparator.comparingInt(FogDevice::getId)) // 必须排序以保证索引一致
                .collect(Collectors.toList());
        if (this.deployableNodes.isEmpty()) {
            throw new IllegalArgumentException("No deployable nodes (Cloud, Gateway, Edge Node) found in the environment!");
        }
        System.out.println("Found " + this.deployableNodes.size() + " deployable nodes.");

        // 确保 API 服务器已启动
//        startRestApiServerOnce();

        // --- 等待 Agent 通过 API 完成一个 Episode 的部署 ---
        System.out.println("RLPlacementLogic run: Waiting for Python Agent to complete deployment via API...");
        episodeLatch = new CountDownLatch(1); // 初始化 Latch
        try {
            // 这里会阻塞，直到 Agent 调用 /get_final_reward 后 Latch 被 countDown
            episodeLatch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("RLPlacementLogic run interrupted while waiting for Agent.");
            // 可能需要返回一个空的或错误的 PlacementLogicOutput
            return new PlacementLogicOutput(new HashMap<>(), new HashMap<>(), new HashMap<>());
        }
        System.out.println("RLPlacementLogic run: Agent finished deployment. Generating output.");

        // --- Agent 完成部署后，根据最终的 currentPlacementMap 生成输出 ---
        return generatePlacementOutput();
    }

    /**
     * 重置内部状态以开始新一轮的部署决策 (Episode).
     * @param requests 初始放置请求
     */
    private void resetInternalState(List<PlacementRequest> requests) {
        currentPlacementMap = new HashMap<>();
        currentCpuLoad = new HashMap<>();
        currentRamLoad = new HashMap<>();
        modulesToPlaceOrder = new ArrayList<>();
        currentModuleIndex = 0;
        currentApp = null; // 重置当前应用

        // 初始化资源负载
        for (FogDevice dev : deployableNodes) {
            currentCpuLoad.put(dev.getId(), 0.0);
            currentRamLoad.put(dev.getId(), 0);
        }

        // 简化处理：假设只处理第一个请求对应的应用
        if (requests == null || requests.isEmpty()) {
            Logger.error("RLPlacementLogic", "No placement requests provided during reset!");
            return;
        }
        PlacementRequest req = requests.get(0); // TODO: 扩展以支持多请求/多应用并行或顺序处理
        currentApp = applicationInfo.get(req.getApplicationId());
        if (currentApp == null) {
            Logger.error("RLPlacementLogic", "Application not found for appId: " + req.getApplicationId());
            return;
        }

        System.out.println("Resetting state for Application: " + currentApp.getAppId());

        // 1. 处理预先放置的模块 (来自 PlacementRequest)
        for (Map.Entry<String, Integer> entry : req.getPlacedMicroservices().entrySet()) {
            String moduleName = entry.getKey();
            Integer nodeId = entry.getValue();
            if (nodeId == null) continue;

            currentPlacementMap.put(moduleName, nodeId);
            AppModule module = getAppModuleByName(currentApp, moduleName);
            FogDevice node = fogDeviceMap.get(nodeId); // 使用 Map 快速查找

            // 仅当预放置节点是可部署节点时，才更新模拟负载
            if (module != null && node != null && deployableNodes.contains(node)) {
                currentCpuLoad.put(nodeId, currentCpuLoad.getOrDefault(nodeId, 0.0) + module.getMips());
                currentRamLoad.put(nodeId, currentRamLoad.getOrDefault(nodeId, 0) + module.getRam());
                System.out.println("  Pre-placed: " + moduleName + " on Node " + nodeId + " (Load updated)");
            } else if (module != null) {
                System.out.println("  Pre-placed: " + moduleName + " on Node " + nodeId + " (Non-deployable node, load not tracked here)");
            }
        }

        // 2. 确定需要 RL 放置的模块及其顺序
        List<String> remainingModules = currentApp.getModules().stream()
                .map(AppModule::getName)
                .filter(name -> !currentPlacementMap.containsKey(name))
                .collect(Collectors.toList());

        // TODO: 实现严格的拓扑排序或基于依赖检查的放置顺序确定
        // 临时方法：使用 getModulesToPlaceLogic 迭代查找顺序 (效率较低但保证依赖)
        Set<String> placedSet = new HashSet<>(currentPlacementMap.keySet());
        while (placedSet.size() < currentApp.getModules().size()) {
            boolean added = false;
            List<String> placeableNow = getModulesToPlaceLogic(placedSet, currentApp);
            for(String module : placeableNow) {
                if (!placedSet.contains(module)) {
                    modulesToPlaceOrder.add(module);
                    placedSet.add(module); // 标记为"待放置"，防止重复添加
                    added = true;
                }
            }
            if (!added && placedSet.size() < currentApp.getModules().size()) {
                // 可能出现循环依赖或无法放置的情况
                Logger.error("RLPlacementLogic", "Cannot determine placement order. Possible circular dependency or error in getModulesToPlaceLogic.");
                // 清空顺序，强制结束
                modulesToPlaceOrder.clear();
                break;
            }
            // 更新 placedSet 以反映新加入待放置列表的模块，以便下一轮依赖检查
            // 注意：这里 placedSet 包含真正已放置和待放置的，逻辑需清晰
            // 更好的方法是单独维护一个 'pendingPlacement' 集合
        }
        // 修正：placedSet 应该只包含真正物理放置的。我们应该在 getModulesToPlaceLogic 中检查 currentPlacementMap
        placedSet = new HashSet<>(currentPlacementMap.keySet()); // 重新获取真正放置的
        modulesToPlaceOrder.clear(); // 清空旧的
        Set<String> addedToOrder = new HashSet<>(); // 防止重复添加
        while(modulesToPlaceOrder.size() < remainingModules.size()) {
            boolean changed = false;
            List<String> canPlaceNow = getModulesToPlaceLogic(new HashSet<>(currentPlacementMap.keySet()), currentApp);
            for (String module : canPlaceNow) {
                if (!currentPlacementMap.containsKey(module) && !addedToOrder.contains(module)) {
                    modulesToPlaceOrder.add(module);
                    addedToOrder.add(module);
                    changed = true;
                }
            }
            // 模拟放置，以便下一轮检查（这很复杂，最好有专门的函数）
            // 简化：假设 getModulesToPlaceLogic 内部能处理，或者就按固定顺序（可能出错）
            if (!changed) { // 如果一轮下来没有任何模块可添加，说明有问题
                Logger.error("RLPlacementLogic", "Stuck determining placement order. Check dependencies for app " + currentApp.getAppId());
                // 随便加一个剩下的，尝试继续 (非常不推荐，仅为示例)
                Optional<String> next = remainingModules.stream().filter(m -> !addedToOrder.contains(m)).findFirst();
                if(next.isPresent()) {
                    modulesToPlaceOrder.add(next.get());
                    addedToOrder.add(next.get());
                } else {
                    break; // 确实没有了
                }
            }

        }


        System.out.println("  Placement Order: " + modulesToPlaceOrder);
        currentModuleIndex = 0;
    }


    /**
     * 构建当前状态的状态向量和动作掩码.
     * @return StateRepresentation 包含向量和掩码
     */
    private StateRepresentation buildStateRepresentation() {
        List<Double> stateVector = calculateStateVector();
        List<Boolean> actionMask = calculateActionMask();
        return new StateRepresentation(stateVector, actionMask);
    }

    /**
     * 计算当前状态的状态向量 S_t.
     * @return 状态向量 (List<Double>)
     */
    private List<Double> calculateStateVector() {
        List<Double> stateVector = new ArrayList<>();

        // A. 节点资源状态 (归一化)
        for (FogDevice node : deployableNodes) {
            double totalMips = node.getHost().getTotalMips();
            int totalRam = node.getHost().getRam(); // 单位 MB
            double usedMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
            int usedRam = currentRamLoad.getOrDefault(node.getId(), 0);

            // 归一化到 [0, 1] 范围 (可用资源比例)
            double availableMipsRatio = (totalMips > 0) ? Math.max(0.0, (totalMips - usedMips) / totalMips) : 0.0;
            double availableRamRatio = (totalRam > 0) ? Math.max(0.0, (double)(totalRam - usedRam) / totalRam) : 0.0;

            stateVector.add(availableMipsRatio);
            stateVector.add(availableRamRatio);
            // TODO: 可以添加更多节点信息，如层级(level)，类型(CLOUD/FON/FCN)等，可能需要编码
            // stateVector.add((double)node.getLevel()); // 示例：添加层级
        }

        // B. 应用部署状态
        int M = (currentApp != null) ? currentApp.getModules().size() : 0;
        List<String> moduleNames = (currentApp != null) ?
                currentApp.getModules().stream().map(AppModule::getName).collect(Collectors.toList()) :
                new ArrayList<>();

        // 部署位置向量 (使用节点索引 + 1，0表示未部署，-1保留?)
        // 为了固定长度，即使应用不同，也需要一个最大模块数 M_max，不足的补位
        // 简化：假设 M 固定或已知上限
        List<Double> placementVec = new ArrayList<>(Collections.nCopies(M, 0.0)); // 0 表示未部署
        // 待部署掩码向量
        List<Double> readyVec = new ArrayList<>(Collections.nCopies(M, 0.0));
        String nextModule = getNextModuleToPlace();
        int nextModuleIdx = (nextModule != null && !moduleNames.isEmpty()) ? moduleNames.indexOf(nextModule) : -1;

        for (int i = 0; i < M; i++) {
            String moduleName = moduleNames.get(i);
            if (currentPlacementMap.containsKey(moduleName)) {
                int nodeId = currentPlacementMap.get(moduleName);
                // 找到 nodeId 在 deployableNodes 中的索引
                int nodeIndex = -1;
                for(int j=0; j < deployableNodes.size(); j++){
                    if(deployableNodes.get(j).getId() == nodeId){
                        nodeIndex = j;
                        break;
                    }
                }
                if (nodeIndex != -1) {
                    placementVec.set(i, (double)(nodeIndex + 1)); // 使用索引+1表示部署位置
                } else {
                    // 部署在了非 deployable 节点 (例如 Client)，特殊标记?
                    placementVec.set(i, -1.0); // 暂定 -1
                }
            }
        }
        if (nextModuleIdx != -1) {
            readyVec.set(nextModuleIdx, 1.0);
        }
        stateVector.addAll(placementVec);
        stateVector.addAll(readyVec);

        // C. 待部署模块需求与交互 (归一化)
        if (nextModule != null && currentApp != null) {
            AppModule module = getAppModuleByName(currentApp, nextModule);
            // TODO: 定义 MIPS, RAM, Volume 的最大预期值用于归一化
            double maxMips = 12000.0; // 例如，取自 edgeNodeCpus 的最大值
            double maxRam = 8192.0;   // 例如，取自 edgeNodeRam 的最大值
            double maxNwLength = 10000.0; // 估计一个网络负载的最大值

            double mipsReq = (module != null) ? (double) module.getMips() : 0.0;
            double ramReq = (module != null) ? (double) module.getRam() : 0.0;
            double volIn = 0;
            double volOut = 0;

            for (AppEdge edge : currentApp.getEdges()) {
                if (edge.getDestination().equals(nextModule) && currentPlacementMap.containsKey(edge.getSource())) {
                    volIn += edge.getTupleNwLength();
                }
                if (edge.getSource().equals(nextModule) && currentPlacementMap.containsKey(edge.getDestination())) {
                    volOut += edge.getTupleNwLength();
                }
            }
            stateVector.add(mipsReq / maxMips);
            stateVector.add(ramReq / maxRam);
            stateVector.add(volIn / maxNwLength); // 归一化交互量
            stateVector.add(volOut / maxNwLength);
        } else {
            // 没有待部署模块，用 0 填充
            stateVector.addAll(Arrays.asList(0.0, 0.0, 0.0, 0.0));
        }

        //System.out.println("State Vector Size: " + stateVector.size()); // 调试用
        //System.out.println("State Vector: " + stateVector);
        return stateVector;
    }

    /**
     * 计算当前状态下合法的动作掩码.
     * @return 动作掩码 (List<Boolean>), 长度等于 deployableNodes 数量
     */
    private List<Boolean> calculateActionMask() {
        int numActions = deployableNodes.size();
        List<Boolean> mask = new ArrayList<>(Collections.nCopies(numActions, false)); // 默认都不可行
        String moduleName = getNextModuleToPlace();

        if (moduleName == null || currentApp == null) {
            // System.out.println("Mask: No module to place or app is null.");
            return mask; // 没有模块需要放置或应用未设置
        }

        AppModule module = getAppModuleByName(currentApp, moduleName);
        if (module == null) {
            Logger.error("RLPlacementLogic", "Module " + moduleName + " not found in app " + currentApp.getAppId());
            return mask; // 模块未找到
        }

        long mipsReq = (long) module.getMips();
        int ramReq = module.getRam(); // 单位 MB

        //System.out.print("Mask for " + moduleName + " (Req: " + mipsReq + " MIPS, " + ramReq + " RAM): [");
        for (int i = 0; i < numActions; i++) {
            FogDevice node = deployableNodes.get(i);
            int nodeId = node.getId();

            double totalMips = node.getHost().getTotalMips();
            int totalRam = node.getHost().getRam();
            double currentMipsLoad = currentCpuLoad.getOrDefault(nodeId, 0.0);
            int currentRamLoadVal = currentRamLoad.getOrDefault(nodeId, 0); // 注意变量名

            if ((totalMips - currentMipsLoad) >= mipsReq && (totalRam - currentRamLoadVal) >= ramReq) {
                mask.set(i, true);
                //System.out.print(" T(" + nodeId + ")");
            } else {
                //System.out.print(" F(" + nodeId + ")");
            }
        }
        //System.out.println(" ]");
        return mask;
    }


    /**
     * 执行 Agent 选择的动作.
     * @param actionNodeIndex Agent 选择的动作，对应 deployableNodes 的索引
     * @return ActionResult 包含下一步的状态表示、即时奖励和完成标志
     */
    private ActionResult executeAction(int actionNodeIndex) {
        String moduleName = getNextModuleToPlace();
        List<Boolean> currentMask = calculateActionMask(); // 获取当前有效动作

        // 检查动作是否有效
        if (moduleName == null || currentApp == null || actionNodeIndex < 0 || actionNodeIndex >= currentMask.size() || !currentMask.get(actionNodeIndex)) {
            System.err.println("Warning: Agent selected an invalid action (" + actionNodeIndex + ") or no module to place.");
            // 保持状态不变，返回惩罚和未完成
            StateRepresentation currentSr = buildStateRepresentation(); // 重新计算当前状态表示
            return new ActionResult(currentSr, penalty_invalid_action, false); // 返回无效动作惩罚
        }

        int nodeId = deployableNodes.get(actionNodeIndex).getId(); // 获取真实 Node ID
        AppModule module = getAppModuleByName(currentApp, moduleName);

        // 更新内部状态: 放置模块，增加资源负载
        currentPlacementMap.put(moduleName, nodeId);
        currentCpuLoad.put(nodeId, currentCpuLoad.getOrDefault(nodeId, 0.0) + module.getMips());
        currentRamLoad.put(nodeId, currentRamLoad.getOrDefault(nodeId, 0) + module.getRam());
        currentModuleIndex++; // 指向下一个待放置模块

        boolean done = isAllPlaced(); // 检查是否所有模块都已放置
        double immediateReward = 0.0; // TODO: 计算即时奖励 (例如 -cost(nodeId))

        // 计算下一步的状态表示 (包含向量和掩码)
        StateRepresentation nextSr = buildStateRepresentation();

        return new ActionResult(nextSr, immediateReward, done);
    }

    /**
     * 检查是否所有需要放置的模块都已放置.
     * @return 如果完成返回 true, 否则 false
     */
    private boolean isAllPlaced() {
        return currentModuleIndex >= modulesToPlaceOrder.size();
    }


    /**
     * 计算最终的加权组合奖励 (在一轮部署完成后调用).
     * @return 最终奖励值
     */
    private double calculateFinalReward() {
        if (currentApp == null || currentPlacementMap == null || currentPlacementMap.size() < modulesToPlaceOrder.size() + (currentPlacementMap.containsKey("client" + currentApp.getAppId())?1:0) ) { // 简易检查是否放置完整
            System.err.println("Calculating final reward but deployment might be incomplete or app is null.");
            return penalty_infeasible; // 如果未完成就计算，视为失败
        }

        double R_comm = 0;
        double R_balance = 0;
        double R_penalty = 0; // 检查资源超限

        // 检查最终资源是否超限 (虽然掩码应该阻止，但做最终校验)
        for (int nodeId : currentCpuLoad.keySet()) {
            FogDevice node = fogDeviceMap.get(nodeId);
            if (node != null && deployableNodes.contains(node)) { // 只检查可部署节点
                if (currentCpuLoad.get(nodeId) > node.getHost().getTotalMips() ||
                        currentRamLoad.get(nodeId) > node.getHost().getRam()) {
                    System.err.println("Error: Final placement exceeds resource limit on node " + nodeId);
                    R_penalty = penalty_infeasible;
                    break; // 一旦发现超限，直接返回惩罚
                }
            }
        }
        if (R_penalty < 0) return R_penalty;


        // 计算通信开销 R_comm (负值)
        for (AppEdge edge : currentApp.getEdges()) {
            String src = edge.getSource();
            String dst = edge.getDestination();
            // 确保边连接的是已部署的模块
            if (currentPlacementMap.containsKey(src) && currentPlacementMap.containsKey(dst)) {
                int srcNodeId = currentPlacementMap.get(src);
                int dstNodeId = currentPlacementMap.get(dst);
                if (srcNodeId != dstNodeId) {
                    double nwLength = edge.getTupleNwLength(); // 数据量
                    double latency = getNetworkLatency(srcNodeId, dstNodeId); // 获取延迟
                    // TODO: 可以考虑带宽因素，计算传输时间而非仅延迟
                    // 简化：使用 数据量 * 延迟 作为通信成本的度量
                    if (latency != Double.MAX_VALUE) {
                        R_comm -= nwLength * latency; // 累加负的通信成本
                    } else {
                        System.err.println("Warning: Could not calculate latency between " + srcNodeId + " and " + dstNodeId);
                        R_comm -= nwLength * 1000; // 无法计算延迟时使用大的惩罚值
                    }
                }
            }
        }

        // 计算负载均衡 R_balance (负值，方差越小越好)
        List<Double> mipsUtils = new ArrayList<>();
        List<Double> ramUtils = new ArrayList<>();
        // 选择用于计算均衡性的节点 (例如，所有 Level 2 Edge Nodes)
        List<FogDevice> targetNodesForBalance = deployableNodes.stream()
                .filter(d -> d.getLevel() == 2) // 只考虑 Edge Nodes
                .collect(Collectors.toList());

        if (!targetNodesForBalance.isEmpty()) {
            for (FogDevice node : targetNodesForBalance) {
                double totalMips = node.getHost().getTotalMips();
                int totalRam = node.getHost().getRam();
                double usedMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
                int usedRam = currentRamLoad.getOrDefault(node.getId(), 0);
                mipsUtils.add(totalMips > 0 ? usedMips / totalMips : 0.0);
                ramUtils.add(totalRam > 0 ? (double) usedRam / totalRam : 0.0);
            }
            double varMips = calculateVariance(mipsUtils);
            double varRam = calculateVariance(ramUtils);
            // 可以对 MIPS 和 RAM 方差赋予不同权重
            R_balance = - (0.5 * varMips + 0.5 * varRam); // 累加负的（平均）方差
        }

        // TODO: 计算可选的 R_latency, R_cost

        // 最终加权奖励
        double finalReward = w_comm * R_comm + w_bal * R_balance + R_penalty;
        // finalReward += w_lat * R_latency + w_cost * R_cost; // 如果计算了可选部分

        // 可以添加一个基准值，使得奖励大部分时候是正数（如果需要）
        // finalReward += 1000; // 示例

        return finalReward;
    }

    /**
     * 根据最终的内部放置映射生成 MicroservicesController 需要的输出格式.
     * @return PlacementLogicOutput
     */
    private PlacementLogicOutput generatePlacementOutput() {
        Map<Integer, Map<Application, List<ModuleLaunchConfig>>> perDevice = new HashMap<>();
        Map<Integer, List<Pair<String, Integer>>> serviceDiscoveryInfo = new HashMap<>(); // TODO: 实现服务发现逻辑
        Map<PlacementRequest, Integer> prStatus = new HashMap<>();

        if (currentApp == null || currentPlacementMap == null || currentPlacementMap.isEmpty()) {
            System.err.println("Cannot generate placement output: currentApp or currentPlacementMap is null/empty.");
            // 需要为所有请求设置失败状态
            for(PlacementRequest pr : placementRequests){
                prStatus.put(pr, 0); // 0 表示失败
            }
            return new PlacementLogicOutput(perDevice, serviceDiscoveryInfo, prStatus);
        }

        Map<Integer, Map<String, Integer>> moduleInstanceCounts = new HashMap<>(); // <NodeId, <ModuleName, Count>>

        // 1. 统计每个节点上每个模块的实例数 (这里简化为1)
        for (Map.Entry<String, Integer> entry : currentPlacementMap.entrySet()) {
            String moduleName = entry.getKey();
            int nodeId = entry.getValue();
            moduleInstanceCounts.computeIfAbsent(nodeId, k -> new HashMap<>())
                    .put(moduleName, 1); // 简化：假设每个模块只有一个实例
        }

        // 2. 构建 perDevice Map
        for (Map.Entry<Integer, Map<String, Integer>> nodeEntry : moduleInstanceCounts.entrySet()) {
            int deviceId = nodeEntry.getKey();
            Map<Application, List<ModuleLaunchConfig>> appMap = new HashMap<>();
            for (Map.Entry<String, Integer> moduleEntry : nodeEntry.getValue().entrySet()) {
                String moduleName = moduleEntry.getKey();
                int instanceCount = moduleEntry.getValue();
                // 需要找到这个 module 属于哪个 app (假设我们只有一个 app)
                AppModule appModule = getAppModuleByName(currentApp, moduleName);
                if (appModule != null) {
                    ModuleLaunchConfig mlc = new ModuleLaunchConfig(appModule, instanceCount);
                    appMap.computeIfAbsent(currentApp, k -> new ArrayList<>()).add(mlc);
                } else {
                    System.err.println("Warning: Could not find AppModule for " + moduleName + " when generating output.");
                }
            }
            if (!appMap.isEmpty()) {
                perDevice.put(deviceId, appMap);
            }
        }

        // 3. 构建 serviceDiscoveryInfo (需要更复杂的逻辑)
        // TODO: 实现 getClientServiceNodeIds 类似的逻辑来填充 serviceDiscoveryInfo

        // 4. 填充 prStatus (假设成功)
        for (PlacementRequest req : placementRequests) {
            // 这里假设我们只处理了第一个请求对应的应用
            if (req.getApplicationId().equals(currentApp.getAppId())) {
                prStatus.put(req, -1); // -1 表示成功放置
            } else {
                prStatus.put(req, 0); // 其他未处理的请求标记为失败 (或未处理)
            }
        }


        return new PlacementLogicOutput(perDevice, serviceDiscoveryInfo, prStatus);
    }


    // --- 辅助方法 ---

    private String getNextModuleToPlace() {
        if (modulesToPlaceOrder != null && currentModuleIndex >= 0 && currentModuleIndex < modulesToPlaceOrder.size()) {
            return modulesToPlaceOrder.get(currentModuleIndex);
        }
        return null;
    }

    private AppModule getAppModuleByName(Application app, String name) {
        if (app == null) return null;
        return app.getModuleByName(name);
    }

    private FogDevice getFogDeviceById(int id) {
        // 优先使用 Map 查找提高效率
        return fogDeviceMap.getOrDefault(id, null);
        /*
        // 备用：遍历查找
        for (FogDevice device : fogDevices) {
            if (device.getId() == id) {
                return device;
            }
        }
        return null;
        */
    }

    private String getAppIdFromModuleName(String moduleName) {
        if (currentApp != null && moduleName.endsWith(currentApp.getAppId())) {
            return currentApp.getAppId();
        }
        // Fallback: 遍历查找 (效率低)
        for (String appId : applicationInfo.keySet()) {
            if (moduleName.endsWith(appId)) {
                return appId;
            }
        }
        Logger.error("RLPlacementLogic", "Could not determine AppId for module: " + moduleName);
        return null; // 或者抛出异常
    }

    // 计算网络延迟 (需要精确实现)
    private double getNetworkLatency(int id1, int id2) {
        if (id1 == id2) return 0;

        FogDevice d1 = getFogDeviceById(id1);
        FogDevice d2 = getFogDeviceById(id2);
        if (d1 == null || d2 == null) {
            System.err.println("Cannot get latency: Node not found (" + id1 + " or " + id2 + ")");
            return Double.MAX_VALUE; // 表示不可达或错误
        }

        // 情况1: 集群内部 (假设 Level 2 且父节点相同)
        if (d1.getLevel() == 2 && d2.getLevel() == 2 && d1.getParentId() == d2.getParentId()) {
            // 需要从 MicroservicePlacement 类访问静态变量 clusterLatency
            return MicroservicePlacement.clusterLatency;
        }

        // 情况2: 父子关系
        if (d1.getParentId() == id2) { // d2 is parent of d1
            return d1.getUplinkLatency();
        }
        if (d2.getParentId() == id1) { // d1 is parent of d2
            return d2.getUplinkLatency();
        }

        // 情况3: 通过共同父节点通信 (例如，同一网关下的两个 Edge Node)
        if (d1.getParentId() == d2.getParentId() && d1.getParentId() != -1) {
            FogDevice parent = getFogDeviceById(d1.getParentId());
            if (parent != null) {
                // 估算：d1 -> parent -> d2
                return d1.getUplinkLatency() + d2.getUplinkLatency(); // 忽略父节点处理延迟
            }
        }

        // 情况4: 跨网关或到云 (需要更复杂的路由逻辑)
        // TODO: 实现基于跳数或固定值的多跳延迟计算
        // 简化：基于层级差异估算 (非常不准确)
        int levelDiff = Math.abs(d1.getLevel() - d2.getLevel());
        if (levelDiff == 1) return 5.0; // 估算相邻层级
        if (levelDiff == 2) return 10.0; // 估算跨一层
        if (d1.getLevel() == 0 || d2.getLevel() == 0) return 100.0; // 估算到云

        System.err.println("Warning: Using rough latency estimation for " + id1 + " <-> " + id2);
        return 20.0; // 默认一个较大的值
    }

    // 计算方差
    private double calculateVariance(List<Double> data) {
        if (data == null || data.size() <= 1) {
            return 0.0;
        }
        double mean = data.stream().mapToDouble(d -> d).filter(Double::isFinite).average().orElse(0.0);
        double variance = data.stream().mapToDouble(d -> Double.isFinite(d) ? (d - mean) * (d - mean) : 0.0)
                .filter(Double::isFinite).sum() / data.size();
        return variance;
    }

    /**
     * 根据已放置模块集合，确定下一步可以放置哪些模块 (满足依赖).
     * @param placedModules 当前已放置模块的名称集合
     * @param app 应用程序对象
     * @return 可以放置的模块名称列表
     */
    private List<String> getModulesToPlaceLogic(Set<String> placedModules, Application app) {
        List<String> modulesToPlace = new ArrayList<>();
        if (app == null) return modulesToPlace;

        for (AppModule module : app.getModules()) {
            String moduleName = module.getName();
            if (placedModules.contains(moduleName)) {
                continue; // 跳过已放置的
            }

            boolean dependenciesMet = true;
            // 检查所有进入该模块的上行边 (UP) 的源是否已放置
            for (AppEdge edge : app.getEdges()) {
                if (edge.getDestination().equals(moduleName) && edge.getDirection() == Tuple.UP) {
                    if (!placedModules.contains(edge.getSource())) {
                        dependenciesMet = false;
                        break;
                    }
                }
                // 检查所有从该模块出去的下行边 (DOWN) 的目标是否已放置
                // (这个检查可能不必要或取决于具体应用逻辑，暂时注释掉)
                 /*
                 if (edge.getSource().equals(moduleName) && edge.getDirection() == Tuple.DOWN) {
                     if (!placedModules.contains(edge.getDestination())) {
                         dependenciesMet = false;
                         break;
                     }
                 }
                 */
            }

            if (dependenciesMet) {
                modulesToPlace.add(moduleName);
            }
        }
        return modulesToPlace;
    }


    // --- MicroservicePlacementLogic 接口的其他方法 ---

    @Override
    public void updateResources(Map<Integer, Map<String, Double>> resourceAvailability) {
        // 在 RL 模式下，资源由内部模拟状态管理，此方法可能不需要做任何事
        // 或者，可以在 Agent 完成部署后，基于最终的 currentPlacementMap 更新一次 resourceAvailability (如果外部需要)
        System.out.println("RLPlacementLogic: updateResources called (currently no-op).");
         /*
         // 示例：基于最终部署更新（如果需要）
         if (isAllPlaced() && currentPlacementMap != null) {
              Map<Integer, Double> finalCpuUsage = new HashMap<>();
              for(String moduleName : currentPlacementMap.keySet()){
                  int nodeId = currentPlacementMap.get(moduleName);
                  Application app = applicationInfo.get(getAppIdFromModuleName(moduleName));
                  AppModule module = getAppModuleByName(app, moduleName);
                  if(module != null && resourceAvailability.containsKey(nodeId)){
                     finalCpuUsage.put(nodeId, finalCpuUsage.getOrDefault(nodeId, 0.0) + module.getMips());
                  }
              }
              for(int nodeId : resourceAvailability.keySet()){
                  if(finalCpuUsage.containsKey(nodeId)){
                      double currentAvailable = resourceAvailability.get(nodeId).getOrDefault(ControllerComponent.CPU, 0.0);
                      // 这里应该是减少可用量，但原始接口意图不明，可能需要调整
                      // resourceAvailability.get(nodeId).put(ControllerComponent.CPU, currentAvailable - finalCpuUsage.get(nodeId));
                  }
              }
         }
         */
    }

    @Override
    public void postProcessing() {
        // 在仿真结束后调用，用于清理资源，例如停止 API 服务器
        System.out.println("RLPlacementLogic postProcessing: Stopping API server.");
//        stopServer();
    }


    // --- POJO 类定义 (用于 API 响应) ---
    // 保持 StateRepresentation, ActionResult, FinalResult 定义不变
    static class StateRepresentation {
        List<Double> stateVector;
        List<Boolean> actionMask;
        // 添加 getter 以便 Gson 序列化
        public List<Double> getStateVector() { return stateVector; }
        public List<Boolean> getActionMask() { return actionMask; }

        StateRepresentation(List<Double> sv, List<Boolean> am) { stateVector = sv; actionMask = am; }
    }
    static class ActionResult {
        StateRepresentation nextStateRepresentation;
        double immediateReward;
        boolean done;
        // 添加 getter
        public StateRepresentation getNextStateRepresentation() { return nextStateRepresentation; }
        public double getImmediateReward() { return immediateReward; }
        public boolean isDone() { return done; }

        ActionResult(StateRepresentation nextSr, double ir, boolean d) { nextStateRepresentation = nextSr; immediateReward = ir; done = d; }
    }
    static class FinalResult {
        double finalReward;
        // 添加 getter
        public double getFinalReward() { return finalReward; }

        FinalResult(double fr) { finalReward = fr; }
    }

}