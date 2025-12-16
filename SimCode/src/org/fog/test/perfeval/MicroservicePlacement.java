package org.fog.test.perfeval;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Comparator; // [新增 Import]

import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.cloudbus.cloudsim.sdn.overbooking.BwProvisionerOverbooking;
import org.cloudbus.cloudsim.sdn.overbooking.PeProvisionerOverbooking;
import org.fog.application.AppEdge;
import org.fog.application.AppLoop;
import org.fog.application.Application;
import org.fog.application.selectivity.FractionalSelectivity;
import org.fog.entities.Actuator;
import org.fog.entities.FogDevice;
import org.fog.entities.FogDeviceCharacteristics;
import org.fog.entities.MicroserviceFogDevice;
import org.fog.entities.PlacementRequest;
import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.placement.MicroservicesController;
import org.fog.placement.PlacementLogicFactory;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.TimeKeeper;
import org.fog.utils.distribution.DeterministicDistribution;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

public class MicroservicePlacement {

    static List<FogDevice> fogDevices = new ArrayList<FogDevice>();
    static List<Sensor> sensors = new ArrayList<Sensor>();
    static List<Actuator> actuators = new ArrayList<Actuator>();

    static int edgeGateways = 4;
    static Integer[] edgeNodesPerGateway = new Integer[]{11, 11, 11, 11};
    // ... (其他静态变量保持不变) ...
    static Integer[] endDevicesPerEdgeNode = new Integer[]{3, 2, 3, 2, 1, 1};
    private static int edgeNodeIndex = 0;
    static boolean heterogeneousEdgeNodes = true;

    // 保持你之前的异构资源定义
//    static Integer[] edgeNodeCpus = new Integer[]{4800, 5000, 5200, 4800, 5500, 6000, 5000, 4800, 5200, 5500};
//    static Integer[] edgeNodeRam = new Integer[]{4096, 8192, 4096, 4096, 8192, 8192, 8192, 4096, 8192, 4096};
    static Integer[] edgeNodeCpus = new Integer[]{
            3500, 3800, 3200, 3500, 4000,
            3500, 3800, 3200, 3500, 4000
    };

    // RAM 也稍微给多点
    static Integer[] edgeNodeRam = new Integer[]{
            4096, 4096, 4096, 4096, 4096,
            4096, 4096, 4096, 4096, 4096
    };
    static Double[] edgeNodeBusyPower = new Double[]{220.0, 230.0, 240.0, 220.0, 250.0, 280.0, 230.0, 220.0, 240.0, 250.0};
    static Double[] edgeNodeIdlePower = new Double[]{110.0, 115.0, 120.0, 110.0, 125.0, 140.0, 115.0, 110.0, 120.0, 125.0};

    static Integer deviceNum = 0;
    public static double edgeToCloudLatency = 100.0;
    public static double gatewayToEdgeNodeLatency = 5.0;
    public static double endDeviceToEdgeNodeLatency = 2.0;
    public static Double clusterLatency = 2.0;
    static boolean trace_flag = false;

    public static void main(String[] args) {
        // ... (Main 方法逻辑保持不变，确保使用了循环创建 40 个应用的代码) ...
        System.out.println(">>> CODE UPDATED: EdgeGateways = " + edgeGateways);
        Log.printLine("Starting MicroservicePlacement...");

        try {
            Log.disable();
            int num_user = 1;
            Calendar calendar = Calendar.getInstance();
            CloudSim.init(num_user, calendar, trace_flag);

            String appId = "A0";
            createFogDevices(1, appId);

            // 读取配置
            List<Map<String, Object>> appParamsList = parseApplicationConfig("D:\\Code\\Microservice-Deployment\\SimCode\\src\\org\\fog\\test\\perfeval\\ApplicationConfig.json");
            if (appParamsList == null || appParamsList.isEmpty()) throw new RuntimeException("Config empty!");

            List<Application> applications = new ArrayList<>();
            List<PlacementRequest> placementRequests = new ArrayList<>();

            // 获取边缘节点并排序
            List<FogDevice> edgeNodes = new ArrayList<>();
            for(FogDevice d : fogDevices) {
                if(d.getName().startsWith("edge-node")) edgeNodes.add(d);
            }
            edgeNodes.sort(Comparator.comparingInt(FogDevice::getId));

            System.out.println("Available Edge Nodes for Clients: " + edgeNodes.size());

            // --- 动态应用创建循环 ---
            int totalAppsToDeploy = 40;
            int appIndex = 0;

            for (int k = 0; k < totalAppsToDeploy; k++) {
                Map<String, Object> appParams = appParamsList.get(k % appParamsList.size());
                String currentAppId = "A" + k;
                int userId = ((Long) appParams.get("userId")).intValue();

                // [关键] 调用新的动态构建方法
                Application app = createApplication(currentAppId, userId, appParams);
                if (app == null) continue; // 跳过无效配置

                applications.add(app);

                // 分配 Client
                if (edgeNodes.isEmpty()) throw new RuntimeException("No edge nodes found!");
                FogDevice clientNode = edgeNodes.get(appIndex % edgeNodes.size());
                int clientNodeId = clientNode.getId();

                System.out.println("Deploying " + currentAppId + " Client to " + clientNode.getName());
                createSensorAndActuator(currentAppId, userId, clientNodeId, app);

                Map<String, Integer> placedMap = new HashMap<>();
                placedMap.put("client", clientNodeId);

                PlacementRequest req = new PlacementRequest(app.getAppId(), 0, clientNodeId, placedMap);
                placementRequests.add(req);

                appIndex++;
            }

            // 初始化 Controller
            MicroservicesController controller = new MicroservicesController(
                    "controller", fogDevices, sensors, applications,
                    new ArrayList<Integer>(), clusterLatency, PlacementLogicFactory.RL_PLACEMENT
            );

            controller.submitPlacementRequests(placementRequests, 0);
            // =================================================================
            System.out.println("DEBUG: >>>>>> EXECUTING MANUAL APP BINDING... <<<<<<");
            int bindCount = 0;

            // 遍历所有传感器
            for (Sensor s : sensors) {
                // 遍历所有应用，寻找匹配的 AppID
                for (Application app : applications) {
                    if (s.getAppId().equals(app.getAppId())) {
                        s.setApp(app); // <--- 关键！这就是修复 NPE 的核心
                        bindCount++;
                        break;
                    }
                }
            }
            // 遍历所有执行器 (防止 Actuator 也报空指针)
            for (Actuator a : actuators) {
                for (Application app : applications) {
                    if (a.getAppId().equals(app.getAppId())) {
                        a.setApp(app);
                        break;
                    }
                }
            }
            System.out.println("DEBUG: >>>>>> SUCCESS! Bound " + bindCount + " sensors. <<<<<<");
            // =================================================================
            TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());
            CloudSim.startSimulation();
            CloudSim.stopSimulation();

            Log.printLine("Simulation finished!");
        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("Unwanted errors happened");
        }
    }

    // === [核心修改] 动态拓扑构建方法 ===
    @SuppressWarnings("unchecked")
    private static Application createApplication(String appId, int userId, Map<String, Object> params) {
        Application application = new Application(appId, userId);

        // 1. 从 Params 中提取微服务列表 (兼容 JSON 数组)
        List<Map<String, Object>> servicesConfig = (List<Map<String, Object>>) params.get("microservices");

        if (servicesConfig == null || servicesConfig.isEmpty()) {
            System.err.println("ERROR: No 'microservices' definition found for " + appId);
            return null;
        }

        // 2. 添加 Client 模块 (固定存在)
        // client_mips 可以从 params 读取，也可以给默认值
        application.addAppModule("client", 128, 500, 100);

        // 3. 循环添加所有微服务模块 (Vertices)
        for (Map<String, Object> svc : servicesConfig) {
            String name = (String) svc.get("name");
            // 处理数值类型转换 (JSON Simple 解析出来通常是 Long)
            int mips = ((Number) svc.getOrDefault("mips", 1000)).intValue();
            int ram = ((Number) svc.getOrDefault("ram", 512)).intValue();
            int bw = ((Number) svc.getOrDefault("bw_out", 1000)).intValue(); // 简化：用 bw_out 代表模块带宽需求

            application.addAppModule(name, ram, mips, bw);
        }

        // 4. 动态构建链路 (Edges)
        // 拓扑顺序: Sensor -> Client -> Svc[0] -> Svc[1] ... -> Svc[N] -> Client -> Actuator

        // 4.1 Sensor -> Client
        application.addAppEdge("sensor", "client", 100, 200, "sensor_data", Tuple.UP, AppEdge.SENSOR);

        // 4.2 Client -> First Service
        Map<String, Object> firstSvc = servicesConfig.get(0);
        String firstSvcName = (String) firstSvc.get("name");
        double firstInBw = ((Number) firstSvc.getOrDefault("bw_in", 1000)).doubleValue();

        String clientToFirstTuple = "c_" + firstSvcName; // 命名规则: c_target
        application.addAppEdge("client", firstSvcName, firstInBw, 1000, clientToFirstTuple, Tuple.UP, AppEdge.MODULE);

        // 映射: sensor_data -> c_first
        application.addTupleMapping("client", "sensor_data", clientToFirstTuple, new FractionalSelectivity(1.0));

        // 4.3 Service -> Service (中间链条)
        for (int i = 0; i < servicesConfig.size() - 1; i++) {
            Map<String, Object> currSvc = servicesConfig.get(i);
            Map<String, Object> nextSvc = servicesConfig.get(i + 1);

            String currName = (String) currSvc.get("name");
            String nextName = (String) nextSvc.get("name");
            double outBw = ((Number) currSvc.getOrDefault("bw_out", 1000)).doubleValue();

            String tupleType = currName + "_" + nextName; // 命名规则: src_dest

            // 确定输入 Tuple 类型 (用于 Selectivity 映射)
            String inputTupleType;
            if (i == 0) {
                inputTupleType = clientToFirstTuple;
            } else {
                String prevName = (String) servicesConfig.get(i - 1).get("name");
                inputTupleType = prevName + "_" + currName;
            }

            application.addAppEdge(currName, nextName, outBw, 1000, tupleType, Tuple.UP, AppEdge.MODULE);
            application.addTupleMapping(currName, inputTupleType, tupleType, new FractionalSelectivity(1.0));
        }

        // 4.4 Last Service -> Client
        Map<String, Object> lastSvc = servicesConfig.get(servicesConfig.size() - 1);
        String lastName = (String) lastSvc.get("name");
        double lastOutBw = ((Number) lastSvc.getOrDefault("bw_out", 1000)).doubleValue();

        String lastToClientTuple = lastName + "_c";

        // 确定最后一个服务的输入 Tuple
        String lastInputTuple;
        if (servicesConfig.size() == 1) {
            lastInputTuple = clientToFirstTuple;
        } else {
            String prevName = (String) servicesConfig.get(servicesConfig.size() - 2).get("name");
            lastInputTuple = prevName + "_" + lastName;
        }

        application.addAppEdge(lastName, "client", lastOutBw, 1000, lastToClientTuple, Tuple.DOWN, AppEdge.MODULE);
        application.addTupleMapping(lastName, lastInputTuple, lastToClientTuple, new FractionalSelectivity(1.0));

        // 4.5 Client -> Actuator
        application.addAppEdge("client", "actuator", 100, 200, "action", Tuple.DOWN, AppEdge.ACTUATOR);
        application.addTupleMapping("client", lastToClientTuple, "action", new FractionalSelectivity(1.0));

        // 5. 构建 Loop (用于时延监控)
        List<String> loopModules = new ArrayList<>();
        loopModules.add("sensor");
        loopModules.add("client");
        for (Map<String, Object> svc : servicesConfig) {
            loopModules.add((String) svc.get("name"));
        }
        loopModules.add("client");
        loopModules.add("actuator");

        final AppLoop appLoop = new AppLoop(loopModules);
        List<AppLoop> loops = new ArrayList<AppLoop>() {{ add(appLoop); }};
        application.setLoops(loops);

        return application;
    }

    // ... (createSensorAndActuator, createFogDevices, parseApplicationConfig 等辅助方法保持不变) ...
    // 为了完整性，下面列出 parseApplicationConfig 以确保其兼容新的 List 结构

    private static List<Map<String, Object>> parseApplicationConfig(String filePath) {
        List<Map<String, Object>> list = new ArrayList<>();
        JSONParser parser = new JSONParser();
        try {
            JSONArray jsonArray = (JSONArray) parser.parse(new FileReader(filePath));
            for (Object obj : jsonArray) {
                JSONObject jsonObject = (JSONObject) obj;
                // JSONSimple 的 JSONObject 实现了 Map 接口，可以直接强转
                list.add(new HashMap<>(jsonObject));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return list;
    }

    // ... (createFogDevices 等方法请保持您原来的实现) ...
//    private static void createSensorAndActuator(String appId, int userId, int parentDeviceId) {
//        Sensor sensor = new Sensor("s-" + appId, "sensor_data", userId, appId, new DeterministicDistribution(5));
//        sensor.setGatewayDeviceId(parentDeviceId);
//        sensor.setLatency(endDeviceToEdgeNodeLatency);
//        sensors.add(sensor);
//
//        Actuator actuator = new Actuator("a-" + appId, userId, appId, "actuator");
//        actuator.setGatewayDeviceId(parentDeviceId);
//        actuator.setLatency(endDeviceToEdgeNodeLatency);
//        actuators.add(actuator);
//    }
    private static void createSensorAndActuator(String appId, int userId, int parentDeviceId, Application app) {
        // 1. 创建 Sensor
        // [修复] TupleType 从 "sensor_data" 改为 "sensor"，以匹配 AppEdge 的 Source Name
        Sensor sensor = new Sensor("s-" + appId, "sensor", userId, appId, new DeterministicDistribution(5));
        sensor.setGatewayDeviceId(parentDeviceId);
        sensor.setLatency(endDeviceToEdgeNodeLatency);

        // 绑定 App (保持之前的修复)
        sensor.setApp(app);

        sensors.add(sensor);

        // 2. 创建 Actuator
        Actuator actuator = new Actuator("a-" + appId, userId, appId, "actuator");
        actuator.setGatewayDeviceId(parentDeviceId);
        actuator.setLatency(endDeviceToEdgeNodeLatency);

        // 绑定 App (保持之前的修复)
        actuator.setApp(app);

        actuators.add(actuator);
    }

    private static void createFogDevices(int userId, String appId) {
        // ... (保持你原来代码中的 createFogDevices 实现，包含异构资源分配逻辑) ...
        // Cloud 节点配置
        FogDevice cloud = createFogDeviceHelper("cloud", 100000, 40000, 10000, 10000, 0, 0.01, 100, 100, MicroserviceFogDevice.CLOUD);
        cloud.setParentId(-1);
        cloud.setLevel(0);
        fogDevices.add(cloud);

        for (int i = 0; i < edgeGateways; i++) {
            FogDevice gateway = createFogDeviceHelper("gateway-" + i, 2800, 4000, 10000, 10000, 1, 0.0, 107, 83, MicroserviceFogDevice.FCN);
            gateway.setParentId(cloud.getId());
            gateway.setUplinkLatency(edgeToCloudLatency);
            gateway.setLevel(1);
            fogDevices.add(gateway);

            int nodesCount = (i < edgeNodesPerGateway.length) ? edgeNodesPerGateway[i] : edgeNodesPerGateway[0];
            for (int j = 0; j < nodesCount; j++) {
                // [修改] 获取对应索引的配置 (CPU, RAM, Power)
                int cpuPos = deviceNum % edgeNodeCpus.length;
                int ramPos = deviceNum % edgeNodeRam.length;
                int pwrPos = deviceNum % edgeNodeBusyPower.length; // 新增能耗索引

                long mips = edgeNodeCpus[cpuPos];
                int ram = edgeNodeRam[ramPos];
                double busyPwr = edgeNodeBusyPower[pwrPos];
                double idlePwr = edgeNodeIdlePower[pwrPos];

                deviceNum++;

                // [修改] 传入动态能耗参数
                FogDevice edge = createFogDeviceHelper("edge-node-" + i + "-" + j, mips, ram, 2000, 2000, 2, 0.0,
                        busyPwr, idlePwr, // 使用异构能耗
                        MicroserviceFogDevice.FCN);
                if (edge == null) {
                    System.out.println("ERROR: Edge node creation failed for " + i + "-" + j);
                    continue;
                }
                edge.setParentId(gateway.getId());
                edge.setUplinkLatency(gatewayToEdgeNodeLatency);
                edge.setLevel(2);
                fogDevices.add(edge);
            }
        }
    }

    private static FogDevice createFogDeviceHelper(String nodeName, long mips, int ram, long upBw, long downBw, int level, double ratePerMips, double busyPower, double idlePower, String deviceType) {
        List<Pe> peList = new ArrayList<Pe>();
        peList.add(new Pe(0, new PeProvisionerOverbooking(mips)));
        int hostId = FogUtils.generateEntityId();
        PowerHost host = new PowerHost(hostId, new RamProvisionerSimple(ram), new BwProvisionerOverbooking(10000), 1000000, peList, new StreamOperatorScheduler(peList), new FogLinearPowerModel(busyPower, idlePower));
        List<Host> hostList = new ArrayList<Host>();
        hostList.add(host);
        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics("x86", "Linux", "Xen", host, 10.0, 3.0, 0.05, 0.001, 0.0);

        try {
            return new MicroserviceFogDevice(nodeName, characteristics, new AppModuleAllocationPolicy(hostList), new LinkedList<Storage>(), 10, upBw, downBw, 0, 0, ratePerMips, deviceType);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}