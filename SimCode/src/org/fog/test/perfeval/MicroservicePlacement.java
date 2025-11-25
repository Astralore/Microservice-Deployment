package org.fog.test.perfeval;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random; // [新增]

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
import org.fog.placement.RLPlacementLogic;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.TimeKeeper;
import org.fog.utils.distribution.DeterministicDistribution; // [新增]
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

public class MicroservicePlacement {

    static List<FogDevice> fogDevices = new ArrayList<FogDevice>();
    static List<Sensor> sensors = new ArrayList<Sensor>();
    static List<Actuator> actuators = new ArrayList<Actuator>();

    // --- [保留原有拓扑定义] ---
    static int edgeGateways = 1;
    // [修改]: 改为 10，以创建 10 个边缘节点供 10 个应用竞争
    static Integer[] edgeNodesPerGateway = new Integer[]{10};
    // [保留]: 原本的 sensor 生成逻辑被 createSensorAndActuator 替代，但保留变量定义不删
    static Integer[] endDevicesPerEdgeNode = new Integer[]{3, 2, 3, 2, 1, 1};

    private static int edgeNodeIndex = 0;
    static boolean heterogeneousEdgeNodes = true;

    // [保留并扩展]: 资源池
    static Integer[] edgeNodeCpus = new Integer[]{2000, 2500, 4000, 2200, 3800, 2000, 3000, 2500, 4200, 2000};
    static Integer[] edgeNodeRam = new Integer[]{2048, 2048, 4096, 2048, 8192, 2048, 4096, 2048, 8192, 2048};

    static Integer deviceNum = 0;

    public static double edgeToCloudLatency = 100.0;
    public static double gatewayToEdgeNodeLatency = 5.0;
    public static double endDeviceToEdgeNodeLatency = 2.0;
    public static Double clusterLatency = 2.0;

    static boolean trace_flag = false;

    public static void main(String[] args) {
        Log.printLine("Starting MicroservicePlacement...");

        try {
            Log.disable();
            int num_user = 1;
            Calendar calendar = Calendar.getInstance();
            CloudSim.init(num_user, calendar, trace_flag);

            String appId = "A0"; // 默认ID

            // 1. 创建物理环境 (使用原有逻辑)
            createFogDevices(1, appId);

            // 2. 读取配置
            List<Map<String, Object>> appParamsList = parseApplicationConfig("D:/Code/Microservice_Deployment/SimCode/src/org/fog/test/perfeval/ApplicationConfig.json");
            if (appParamsList == null || appParamsList.isEmpty()) throw new RuntimeException("Config empty!");

            List<Application> applications = new ArrayList<>();
            List<PlacementRequest> placementRequests = new ArrayList<>();

            // 获取边缘节点列表 (用于放置 Client)
            List<FogDevice> edgeNodes = new ArrayList<>();
            for(FogDevice d : fogDevices) {
                // 这里的判断逻辑是根据名字，保持简单
                if(d.getName().startsWith("edge-node")) edgeNodes.add(d);
            }

            Random rand = new Random();

            // 3. [修改] 批量创建应用 (A0-A9)
            for (Map<String, Object> appParams : appParamsList) {
                String currentAppId = (String) appParams.get("appId");
                int userId = ((Long) appParams.get("userId")).intValue();

                // 创建 3级微服务链应用
                Application app = createApplication(currentAppId, userId, appParams);
                applications.add(app);

                // 随机选一个边缘节点放 Client
                FogDevice clientNode = edgeNodes.get(rand.nextInt(edgeNodes.size()));
                int clientNodeId = clientNode.getId();

                // [新增] 必须创建 Sensor/Actuator
                createSensorAndActuator(currentAppId, userId, clientNodeId);

                Map<String, Integer> placedMap = new HashMap<>();
                placedMap.put("client", clientNodeId);

                // [修复] PlacementRequest 构造函数传入 int 类型的 gatewayId
                PlacementRequest req = new PlacementRequest(app.getAppId(), 0, clientNodeId, placedMap);
                placementRequests.add(req);

                System.out.println("Initialized " + currentAppId + " on " + clientNode.getName());
            }

            // 4. [修复] 初始化 Controller
            MicroservicesController controller = new MicroservicesController(
                    "controller",
                    fogDevices,
                    sensors,
                    applications, // 第4个参数是 applications List
                    new ArrayList<Integer>(), // clusterLevels
                    clusterLatency,
                    PlacementLogicFactory.RL_PLACEMENT // 使用 RL 策略 ID
            );

            // 提交所有请求
            controller.submitPlacementRequests(placementRequests, 0);

            // 5. 运行
            TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());
            CloudSim.startSimulation();
            CloudSim.stopSimulation();

            Log.printLine("Simulation finished!");
        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("Unwanted errors happened");
        }
    }

    // [新增] 辅助方法：创建 Sensor 和 Actuator
    private static void createSensorAndActuator(String appId, int userId, int parentDeviceId) {
        // Sensor 产生数据
        Sensor sensor = new Sensor("s-" + appId, "sensor_data", userId, appId, new DeterministicDistribution(5));
        sensor.setGatewayDeviceId(parentDeviceId);
        sensor.setLatency(endDeviceToEdgeNodeLatency);
        sensors.add(sensor);

        // Actuator 接收数据
        Actuator actuator = new Actuator("a-" + appId, userId, appId, "actuator");
        actuator.setGatewayDeviceId(parentDeviceId);
        actuator.setLatency(endDeviceToEdgeNodeLatency);
        actuators.add(actuator);
    }

    // [修改] 创建应用逻辑：支持 3 级链条
    private static Application createApplication(String appId, int userId, Map<String, Object> params) {
        Application application = new Application(appId, userId);

        int m1Mips = ((Long) params.getOrDefault("mService1_mips", 2500L)).intValue();
        int m2Mips = ((Long) params.getOrDefault("mService2_mips", 3500L)).intValue();
        // 读取 mService3，默认 2000
        int m3Mips = ((Long) params.getOrDefault("mService3_mips", 2000L)).intValue();

        application.addAppModule("client", 128, 500, 100);
        application.addAppModule("mService1", 1024, m1Mips, 1000);
        application.addAppModule("mService2", 2048, m2Mips, 1000);
        application.addAppModule("mService3", 1024, m3Mips, 1000); // 新增

        // 边: Sensor -> Client -> m1 -> m2 -> m3 -> Client -> Actuator
        application.addAppEdge("sensor", "client", 100, 200, "sensor_data", Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("client", "mService1", 2000, 1000, "c_m1", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("mService1", "mService2", 2500, 1000, "m1_m2", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("mService2", "mService3", 2000, 1000, "m2_m3", Tuple.UP, AppEdge.MODULE); // 新增
        application.addAppEdge("mService3", "client", 1000, 500, "m3_c", Tuple.DOWN, AppEdge.MODULE); // 新增
        application.addAppEdge("client", "actuator", 100, 200, "action", Tuple.DOWN, AppEdge.ACTUATOR);

        // 映射
        application.addTupleMapping("client", "sensor_data", "c_m1", new FractionalSelectivity(1.0));
        application.addTupleMapping("mService1", "c_m1", "m1_m2", new FractionalSelectivity(1.0));
        application.addTupleMapping("mService2", "m1_m2", "m2_m3", new FractionalSelectivity(1.0));
        application.addTupleMapping("mService3", "m2_m3", "m3_c", new FractionalSelectivity(1.0));
        application.addTupleMapping("client", "m3_c", "action", new FractionalSelectivity(1.0));

        final AppLoop loop1 = new AppLoop(new ArrayList<String>() {{
            add("sensor"); add("client"); add("mService1"); add("mService2"); add("mService3"); add("client"); add("actuator");
        }});
        List<AppLoop> loops = new ArrayList<AppLoop>() {{ add(loop1); }};
        application.setLoops(loops);

        return application;
    }

    // [修改] 使用 createFogDeviceHelper 适配新构造函数，同时保留原有拓扑循环
    private static void createFogDevices(int userId, String appId) {
        // Cloud
        FogDevice cloud = createFogDeviceHelper("cloud", 100000, 40000, 10000, 10000, 0, 0.01, 100, 100, MicroserviceFogDevice.CLOUD);
        cloud.setParentId(-1);
        cloud.setLevel(0);
        fogDevices.add(cloud);

        // [保持原结构] 循环创建 Gateway 和 EdgeNodes
        for (int i = 0; i < edgeGateways; i++) {
            FogDevice gateway = createFogDeviceHelper("gateway-" + i, 20000, 8000, 10000, 10000, 1, 0.0, 107, 83, MicroserviceFogDevice.FON);
            gateway.setParentId(cloud.getId());
            gateway.setUplinkLatency(edgeToCloudLatency);
            gateway.setLevel(1);
            fogDevices.add(gateway);

            int nodesCount = (i < edgeNodesPerGateway.length) ? edgeNodesPerGateway[i] : edgeNodesPerGateway[0];
            for (int j = 0; j < nodesCount; j++) {
                int cpuPos = deviceNum % edgeNodeCpus.length;
                int ramPos = deviceNum % edgeNodeRam.length;
                long mips = edgeNodeCpus[cpuPos];
                int ram = edgeNodeRam[ramPos];
                deviceNum++;

                FogDevice edge = createFogDeviceHelper("edge-node-" + i + "-" + j, mips, ram, 2000, 2000, 2, 0.0, 107, 83, MicroserviceFogDevice.FCN);
                edge.setParentId(gateway.getId());
                edge.setUplinkLatency(gatewayToEdgeNodeLatency);
                edge.setLevel(2);
                fogDevices.add(edge);
            }
        }
    }

    // [修复] 适配 MicroserviceFogDevice 构造函数
    // 原名 createFogDevice 改为 createFogDeviceHelper 避免混淆
    private static FogDevice createFogDeviceHelper(String nodeName, long mips, int ram, long upBw, long downBw, int level, double ratePerMips, double busyPower, double idlePower, String deviceType) {
        List<Pe> peList = new ArrayList<Pe>();
        peList.add(new Pe(0, new PeProvisionerOverbooking(mips)));
        int hostId = FogUtils.generateEntityId();
        PowerHost host = new PowerHost(hostId, new RamProvisionerSimple(ram), new BwProvisionerOverbooking(10000), 1000000, peList, new StreamOperatorScheduler(peList), new FogLinearPowerModel(busyPower, idlePower));
        List<Host> hostList = new ArrayList<Host>();
        hostList.add(host);
        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics("x86", "Linux", "Xen", host, 10.0, 3.0, 0.05, 0.001, 0.0);

        try {
            // 这里的 0 是 clusterLinkBandwidth，0 是 uplinkLatency (外部设置)
            return new MicroserviceFogDevice(nodeName, characteristics, new AppModuleAllocationPolicy(hostList), new LinkedList<Storage>(), 10, upBw, downBw, 0, 0, ratePerMips, deviceType);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static List<Map<String, Object>> parseApplicationConfig(String filePath) {
        List<Map<String, Object>> list = new ArrayList<>();
        JSONParser parser = new JSONParser();
        try {
            JSONArray jsonArray = (JSONArray) parser.parse(new FileReader(filePath));
            for (Object obj : jsonArray) {
                JSONObject jsonObject = (JSONObject) obj;
                list.add(new HashMap<>(jsonObject));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return list;
    }
}