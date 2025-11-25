package org.fog.test.perfeval;

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
import org.fog.entities.*;
import org.fog.entities.MicroserviceFogDevice;
import org.fog.entities.PlacementRequest;
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
import org.json.simple.parser.ParseException;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * 仿真环境：针对云边协同的微服务应用
 *
 * 该仿真环境修改自原有的iFogSim微服务示例，以更好地匹配“云-边-端”三层架构.
 * * 1.  **拓扑结构**: 清晰地定义了云中心、边缘层（由网关和计算节点组成）和终端设备层.
 * 2.  **资源异构性**: 云、边、端设备拥有显著不同的计算能力和网络延迟.
 * 3.  **可扩展性**: 您可以轻松地通过修改顶部的静态配置变量来调整拓扑规模.
 * 4.  **算法接口**: 预留了接口用于替换和测试您自己的微服务部署算法.
 *
 * @author Samodha Pallewatta (Original)
 * @author Gemini (Adapted for Cloud-Edge Collaboration)
 */
public class MicroservicePlacementBasic {

    static List<FogDevice> fogDevices = new ArrayList<FogDevice>();
    static List<Sensor> sensors = new ArrayList<Sensor>();
    static List<Actuator> actuators = new ArrayList<Actuator>();

    // ------------------- 云-边-端 拓扑结构配置 -------------------

    // **第一层: 云中心**
    // 云节点总是存在且唯一.

    // **第二层: 边缘层**
    // 边缘层由 "边缘网关" 和 "边缘计算节点" 组成，模拟边缘集群
    static int edgeGateways = 1; // 边缘网关的数量
    static Integer[] edgeNodesPerGateway = new Integer[]{6}; // 每个边缘网关下有多少个计算节点

    // **第三层: 终端层**
    static Integer[] endDevicesPerEdgeNode = new Integer[]{3, 2, 3, 2, 1, 1}; // 每个边缘计算节点下连接多少个终端设备 
    private static int edgeNodeIndex = 0; // 用于追踪正在添加终端设备的边缘节点

    // 边缘节点资源异构性配置
    static boolean heterogeneousEdgeNodes = true; // 是否让边缘节点的资源不同
    static Integer[] edgeNodeCpus = new Integer[]{8000, 12000}; // 边缘节点的CPU资源 (MIPS)
    static Integer[] edgeNodeRam = new Integer[]{6144, 8192};   // 边缘节点的内存资源 (MB)
    static Integer deviceNum = 0; // 用于轮换选择资源

    // 网络延迟配置 (单位: ms)
    static double edgeToCloudLatency = 100.0;       // 边缘网关到云的延迟
    static double gatewayToEdgeNodeLatency = 5.0;   // 边缘网关到其下计算节点的延迟
    static double endDeviceToEdgeNodeLatency = 2.0; // 终端设备到其父边缘节点的延迟
    static Double clusterLatency = 2.0;             // 边缘集群内部节点间的通信延迟
    // ----------------------------------------------------------------

    // 应用相关配置
    static List<Application> applications = new ArrayList<>();
    static int appCount = 1;
    static int appNum = 0;

    public static void main(String[] args) {

        try {

            Log.disable();
            int num_user = 1;
            Calendar calendar = Calendar.getInstance();
            boolean trace_flag = false;

            CloudSim.init(num_user, calendar, trace_flag);

            FogBroker broker = new FogBroker("broker");

            // 1. 从JSON文件加载微服务应用定义
            String fileName = "src/org/fog/test/perfeval/ApplicationConfigBasic" +".json";
            applications = generateAppsFromFile(fileName);
            appNum = 0;

            // 2. 创建云-边-端三层物理拓扑
            createCloudEdgeEndDeviceTopology(broker.getId());

            // 3. 设置和初始化微服务控制器
            // 指定边缘计算节点层 (level=2)
            List<Integer> clusterLevelIdentifier = new ArrayList<>();
            clusterLevelIdentifier.add(2);

            List<Application> appList = new ArrayList<>();
            for (Application application : applications)
                appList.add(application);

            // 部署算法
            // 在 PlacementLogicFactory 中添加在此处引用
            // public static final int MY_CUSTOM_ALGORITHM = 5;
            // 然后在这里使用: int placementAlgo = PlacementLogicFactory.MY_CUSTOM_ALGORITHM;
            int placementAlgo = PlacementLogicFactory.CLUSTERED_MICROSERVICES_PLACEMENT; // 使用默认算法

            MicroservicesController microservicesController = new MicroservicesController("controller", fogDevices, sensors, appList, clusterLevelIdentifier, clusterLatency, placementAlgo);

            // 4. 生成来自终端设备的部署请求
            List<PlacementRequest> placementRequests = new ArrayList<>();
            for (Sensor s : sensors) {
                Map<String, Integer> placedMicroservicesMap = new HashMap<>();
                //  "client" 模块部署在终端设备所连接的父节点上
                placedMicroservicesMap.put("client" + s.getAppId(), s.getGatewayDeviceId());
                PlacementRequest p = new PlacementRequest(s.getAppId(), s.getId(), s.getGatewayDeviceId(), placedMicroservicesMap);
                placementRequests.add(p);
            }
            microservicesController.submitPlacementRequests(placementRequests, 0);

            // 5. 启动仿真
            TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());
            CloudSim.startSimulation();
            CloudSim.stopSimulation();
            Log.printLine("Cloud-Edge Simulation finished!");
        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("Unwanted errors happen");
        }
    }

    /**
     * 创建“云-边-端”三层拓扑结构.
     * @param userId
     */
    private static void createCloudEdgeEndDeviceTopology(int userId) {
        // --- 第一层: 创建云中心 (Level 0) ---
        // 云具有非常高的计算资源和存储，但访问延迟也高
        FogDevice cloud = createFogDevice("cloud", 1000000, 65536, 10000, 10000, 0, 0.01, 1000, 800, MicroserviceFogDevice.CLOUD);
        cloud.setParentId(-1);
        fogDevices.add(cloud);

        // --- 第二层: 创建边缘层 (Level 1 和 2) ---
        // 边缘层由“网关”和“计算节点”组成，模拟一个地理上集中的边缘集群
        for (int i = 0; i < edgeGateways; i++) {
            // 创建边缘网关 (Level 1)
            FogDevice gateway = createFogDevice("edge-gateway-" + i, 20000, 16384, 5000, 5000, 1, 0.0, 107.339, 83.4333, MicroserviceFogDevice.FON);
            gateway.setParentId(cloud.getId());
            gateway.setUplinkLatency(edgeToCloudLatency); // 设置边缘到云的延迟
            fogDevices.add(gateway);

            // 在此网关下创建多个边缘计算节点 (Level 2)
            for (int j = 0; j < edgeNodesPerGateway[i]; j++) {
                addEdgeNode(String.valueOf(j), userId, gateway.getId());
            }
        }
    }

    /**
     * 创建一个边缘计算节点并连接终端设备.
     * @param id 节点ID
     * @param userId 用户ID
     * @param parentId 父节点ID (即边缘网关ID)
     * @return
     */
    private static FogDevice addEdgeNode(String id, int userId, int parentId) {
        FogDevice edgeNode;
        // 根据配置决定是否创建异构资源
        if (heterogeneousEdgeNodes) {
            int pos = deviceNum % edgeNodeCpus.length;
            edgeNode = createFogDevice("edge-node-" + id, edgeNodeCpus[pos], edgeNodeRam[pos], 3000, 3000, 2, 0.0, 107.339, 83.4333, MicroserviceFogDevice.FCN);
            deviceNum++;
        } else {
            // 创建同构资源
            edgeNode = createFogDevice("edge-node-" + id, 10000, 8192, 3000, 3000, 2, 0.0, 107.339, 83.4333, MicroserviceFogDevice.FCN);
        }

        fogDevices.add(edgeNode);
        edgeNode.setParentId(parentId);
        edgeNode.setUplinkLatency(gatewayToEdgeNodeLatency); // 网关到计算节点的延迟

        // --- 第三层: 创建终端设备 (Level 3) 并连接到此边缘节点 ---
        int connectedEndDevices = endDevicesPerEdgeNode[edgeNodeIndex];
        for (int i = 0; i < connectedEndDevices; i++) {
            String mobileId = id + "-" + i;
            // 将终端设备（Mobile）添加到此边缘节点下
            FogDevice mobile = addEndDevice(mobileId, userId, edgeNode.getId());
            mobile.setUplinkLatency(endDeviceToEdgeNodeLatency); // 终端到边缘的延迟
            fogDevices.add(mobile);
        }
        edgeNodeIndex++;
        return edgeNode;
    }

    /**
     * 创建一个终端设备 (例如手机、摄像头).
     * @param id 终端ID
     * @param userId 用户ID
     * @param parentId 父节点ID (即边缘计算节点ID)
     * @return
     */
    private static FogDevice addEndDevice(String id, int userId, int parentId) {
        Application application = applications.get(appNum % appCount);
        String appId = application.getAppId();
        double throughput = 200;

        // 终端设备资源较弱 (Level 3)
        FogDevice mobile = createFogDevice("m-" + id, 1000, 2048, 500, 500, 3, 0, 87.53, 82.44, MicroserviceFogDevice.CLIENT);
        mobile.setParentId(parentId);

        // 创建与终端设备相连的传感器和执行器
        Sensor eegSensor = new Sensor("s-" + id, "sensor" + appId, userId, appId, new DeterministicDistribution(1000 / (throughput / 9 * 10)));
        eegSensor.setApp(application);
        sensors.add(eegSensor);

        Actuator display = new Actuator("a-" + id, userId, appId, "actuator" + appId);
        actuators.add(display);

        // 将传感器和执行器连接到它们的“网关”，即此终端设备
        eegSensor.setGatewayDeviceId(mobile.getId());
        eegSensor.setLatency(5.0);  // 传感器到终端的内部延迟

        display.setGatewayDeviceId(mobile.getId());
        display.setLatency(1.0);  // 终端到执行器的内部延迟
        display.setApp(application);

        appNum++;
        return mobile;
    }

    // ------------------- 以下为辅助方法 -------------------

    private static MicroserviceFogDevice createFogDevice(String nodeName, long mips,
                                                         int ram, long upBw, long downBw, int level, double ratePerMips, double busyPower, double idlePower, String deviceType) {
        List<Pe> peList = new ArrayList<Pe>();
        peList.add(new Pe(0, new PeProvisionerOverbooking(mips)));
        int hostId = FogUtils.generateEntityId();
        long storage = 1000000;
        int bw = 10000;
        PowerHost host = new PowerHost(hostId, new RamProvisionerSimple(ram), new BwProvisionerOverbooking(bw), storage, peList, new StreamOperatorScheduler(peList), new FogLinearPowerModel(busyPower, idlePower));
        List<Host> hostList = new ArrayList<Host>();
        hostList.add(host);
        String arch = "x86";
        String os = "Linux";
        String vmm = "Xen";
        double time_zone = 10.0;
        double cost = 3.0;
        double costPerMem = 0.05;
        double costPerStorage = 0.001;
        double costPerBw = 0.0;
        LinkedList<Storage> storageList = new LinkedList<Storage>();
        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics(arch, os, vmm, host, time_zone, cost, costPerMem, costPerStorage, costPerBw);
        MicroserviceFogDevice fogdevice = null;
        try {
            fogdevice = new MicroserviceFogDevice(nodeName, characteristics, new AppModuleAllocationPolicy(hostList), storageList, 10, upBw, downBw, 1250000, 0, ratePerMips, deviceType);
        } catch (Exception e) {
            e.printStackTrace();
        }
        fogdevice.setLevel(level);
        return fogdevice;
    }

    private static List<Application> generateAppsFromFile(String fileName) {
        List<Application> apps = new ArrayList<>();
        JSONParser jsonParser = new JSONParser();
        try (FileReader reader = new FileReader(fileName)) {
            Object obj = jsonParser.parse(reader);
            JSONArray appParamList = (JSONArray) obj;
            for (int i = 0; i < appParamList.size(); i++) {
                apps.add(createApplication((JSONObject) appParamList.get(i)));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        }
        return apps;
    }

    @SuppressWarnings({"serial"})
    private static Application createApplication(JSONObject applicationParameters) {
        String appId = (String) applicationParameters.get("appId");
        int userId = Math.toIntExact((long) applicationParameters.get("userId"));
        Application application = Application.createApplication(appId, userId);
        String client = "client" + appId;
        String mService1 = "mService1" + appId;
        String mService2 = "mService2" + appId;
        String sensor = "sensor" + appId;
        String actuator = "actuator" + appId;
        application.addAppEdge(sensor, client, 1000, (Double) applicationParameters.get("nwLength"), sensor, Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge(client, mService1, (Double) applicationParameters.get("cpu_c_m1"), (Double) applicationParameters.get("nw_c_m1"), "c_m1" + appId, Tuple.UP, AppEdge.MODULE);
        application.addAppEdge(mService1, mService2, (Double) applicationParameters.get("cpu_m1_m2"), (Double) applicationParameters.get("nw_m1_m2"), "m1_m2" + appId, Tuple.UP, AppEdge.MODULE);
        application.addAppEdge(mService2, client, 28, 200, "m2_c" + appId, Tuple.DOWN, AppEdge.MODULE);
        application.addAppEdge(client, actuator, 28, 200, "a_m2c" + appId, Tuple.DOWN, AppEdge.ACTUATOR);
        application.addAppEdge(client, actuator, 28, 200, "a_m3c" + appId, Tuple.DOWN, AppEdge.ACTUATOR);
        application.addAppModule(client, 128, Math.toIntExact((long) applicationParameters.get("client")), 100);
        application.addAppModule(mService1, 512, Math.toIntExact((long) applicationParameters.get("mService1")), 200);
        application.addAppModule(mService2, 512, Math.toIntExact((long) applicationParameters.get("mService2")), 200);
        application.addTupleMapping(client, sensor, "c_m1" + appId, new FractionalSelectivity(0.9));
        application.addTupleMapping(client, "m2_c" + appId, "a_m2c" + appId, new FractionalSelectivity(1.0));
        application.addTupleMapping(client, "m3_c" + appId, "a_m3c" + appId, new FractionalSelectivity(1.0));
        application.addTupleMapping(mService1, "c_m1" + appId, "m1_m2" + appId, new FractionalSelectivity(1.0));
        application.addTupleMapping(mService1, "c_m1" + appId, "m1_m3" + appId, new FractionalSelectivity(1.0));
        application.addTupleMapping(mService2, "m1_m2" + appId, "m2_c" + appId, new FractionalSelectivity(1.0));
        final AppLoop loop1 = new AppLoop(new ArrayList<String>() {{
            add(sensor);
            add(client);
            add(mService1);
            add(mService2);
            add(client);
            add(actuator);
        }});
        List<AppLoop> loops = new ArrayList<AppLoop>() {{
            add(loop1);
        }};
        application.setLoops(loops);
        return application;
    }
}
