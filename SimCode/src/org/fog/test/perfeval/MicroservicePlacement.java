package org.fog.test.perfeval;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

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
    static Integer[] endDevicesPerEdgeNode = new Integer[]{3, 2, 3, 2, 1, 1};

    private static int edgeNodeIndex = 0;
    static boolean heterogeneousEdgeNodes = true;

    static Integer[] edgeNodeCpus = new Integer[]{
            4800, 5000, 5200, 4800, 5500,
            6000, 5000, 4800, 5200, 5500
    };

    // RAM 也相应给足，避免内存成为瓶颈干扰 CPU 调度的实验结论
    static Integer[] edgeNodeRam = new Integer[]{
            4096, 8192, 4096, 4096, 8192,
            8192, 8192, 4096, 8192, 4096
    };

    // 高性能 = 高能耗，迫使 RL 学会权衡
    static Double[] edgeNodeBusyPower = new Double[]{
            220.0, 230.0, 240.0, 220.0, 250.0,
            280.0, 230.0, 220.0, 240.0, 250.0
    };

    // 待机功耗 (通常是满载的 50%-60%)
    static Double[] edgeNodeIdlePower = new Double[]{
            110.0, 115.0, 120.0, 110.0, 125.0,
            140.0, 115.0, 110.0, 120.0, 125.0
    };

    static Integer deviceNum = 0;

    public static double edgeToCloudLatency = 100.0;
    public static double gatewayToEdgeNodeLatency = 5.0;
    public static double endDeviceToEdgeNodeLatency = 2.0;
    public static Double clusterLatency = 2.0;

    static boolean trace_flag = false;

    public static void main(String[] args) {
        System.out.println(">>> CODE UPDATED: EdgeGateways = " + edgeGateways);
        Log.printLine("Starting MicroservicePlacement...");

        try {
            Log.disable();
            int num_user = 1;
            Calendar calendar = Calendar.getInstance();
            CloudSim.init(num_user, calendar, trace_flag);

            String appId = "A0";

            // 1. 创建物理环境
            createFogDevices(1, appId);

            // 2. 读取配置
            List<Map<String, Object>> appParamsList = parseApplicationConfig("D:/Code/Microservice_Deployment/SimCode/src/org/fog/test/perfeval/ApplicationConfig.json");
            if (appParamsList == null || appParamsList.isEmpty()) throw new RuntimeException("Config empty!");

            List<Application> applications = new ArrayList<>();
            List<PlacementRequest> placementRequests = new ArrayList<>();

            // 获取边缘节点列表并排序，保证分配顺序确定
            List<FogDevice> edgeNodes = new ArrayList<>();
            for(FogDevice d : fogDevices) {
                if(d.getName().startsWith("edge-node")) edgeNodes.add(d);
            }
            edgeNodes.sort((a, b) -> Integer.compare(a.getId(), b.getId()));

            System.out.println("Available Edge Nodes for Clients: " + edgeNodes.size());

            // 3. 批量创建应用 (A0-A9)
            // [修复] 使用索引进行轮询分配，防止负载不均
            int appIndex = 0;

            for (Map<String, Object> appParams : appParamsList) {
                String currentAppId = (String) appParams.get("appId");
                int userId = ((Long) appParams.get("userId")).intValue();

                // 创建应用
                Application app = createApplication(currentAppId, userId, appParams);
                applications.add(app);

                // [修复逻辑] 强制均匀分配：App 0 -> Node 0, App 1 -> Node 1 ...
                if (edgeNodes.isEmpty()) throw new RuntimeException("No edge nodes found!");

                FogDevice clientNode = edgeNodes.get(appIndex % edgeNodes.size());
                int clientNodeId = clientNode.getId();

                System.out.println("Deploying " + currentAppId + " Client/Sensor to " + clientNode.getName() + " (ID: " + clientNodeId + ")");

                // 创建 Sensor/Actuator 并绑定到该边缘节点
                createSensorAndActuator(currentAppId, userId, clientNodeId);

                Map<String, Integer> placedMap = new HashMap<>();
                placedMap.put("client", clientNodeId);

                PlacementRequest req = new PlacementRequest(app.getAppId(), 0, clientNodeId, placedMap);
                placementRequests.add(req);

                // 索引递增
                appIndex++;
            }
            System.out.println("DEBUG: fogDevices list size in Main = " + fogDevices.size());
            // 4. 初始化 Controller
            MicroservicesController controller = new MicroservicesController(
                    "controller",
                    fogDevices,
                    sensors,
                    applications,
                    new ArrayList<Integer>(),
                    clusterLatency,
                    PlacementLogicFactory.RL_PLACEMENT
            );

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

    private static void createSensorAndActuator(String appId, int userId, int parentDeviceId) {
        Sensor sensor = new Sensor("s-" + appId, "sensor_data", userId, appId, new DeterministicDistribution(5));
        sensor.setGatewayDeviceId(parentDeviceId);
        sensor.setLatency(endDeviceToEdgeNodeLatency);
        sensors.add(sensor);

        Actuator actuator = new Actuator("a-" + appId, userId, appId, "actuator");
        actuator.setGatewayDeviceId(parentDeviceId);
        actuator.setLatency(endDeviceToEdgeNodeLatency);
        actuators.add(actuator);
    }

    private static Application createApplication(String appId, int userId, Map<String, Object> params) {
        Application application = new Application(appId, userId);

        int m1Mips = ((Long) params.getOrDefault("mService1_mips", 1000L)).intValue();
        int m2Mips = ((Long) params.getOrDefault("mService2_mips", 1500L)).intValue();
        int m3Mips = ((Long) params.getOrDefault("mService3_mips", 1000L)).intValue();

        System.out.println(String.format("DEBUG: App %s Created with MIPS: m1=%d, m2=%d, m3=%d", appId, m1Mips, m2Mips, m3Mips));

        application.addAppModule("client", 128, 500, 100);
        application.addAppModule("mService1", 1024, m1Mips, 1000);
        application.addAppModule("mService2", 2048, m2Mips, 1000);
        application.addAppModule("mService3", 1024, m3Mips, 1000);

        application.addAppEdge("sensor", "client", 100, 200, "sensor_data", Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("client", "mService1", 2000, 1000, "c_m1", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("mService1", "mService2", 2500, 1000, "m1_m2", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("mService2", "mService3", 2000, 1000, "m2_m3", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("mService3", "client", 1000, 500, "m3_c", Tuple.DOWN, AppEdge.MODULE);
        application.addAppEdge("client", "actuator", 100, 200, "action", Tuple.DOWN, AppEdge.ACTUATOR);

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

    private static void createFogDevices(int userId, String appId) {
        // Cloud 节点配置
        FogDevice cloud = createFogDeviceHelper("cloud", 100000, 40000, 10000, 10000, 0, 0.01, 100, 100, MicroserviceFogDevice.CLOUD);
        cloud.setParentId(-1);
        cloud.setLevel(0);
        fogDevices.add(cloud);

        for (int i = 0; i < edgeGateways; i++) {
            FogDevice gateway = createFogDeviceHelper("gateway-" + i, 2800, 4000, 10000, 10000, 1, 0.0, 107, 83, MicroserviceFogDevice.FON);
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
            System.out.println("DEBUG: Finished Gateway-" + i + " loop. List size: " + fogDevices.size());
        }
        System.out.println("DEBUG: createFogDevices Finished. Final List size: " + fogDevices.size());
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