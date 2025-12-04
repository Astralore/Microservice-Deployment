package org.fog.placement;

import org.apache.commons.math3.util.Pair;
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
 * 强化学习部署逻辑类
 */
public class RLPlacementLogic implements MicroservicePlacementLogic {

    // --- [配置] 必须与 Python config.py 一致 ---
    private static final int MAX_NODES = 50;
    private static final int API_PORT = 4567;

    // --- 成员变量 ---
    private List<FogDevice> fogDevices;
    private List<FogDevice> deployableNodes; // 仅包含 Cloud, Gateway, Edge
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
        this.fogDevices = fogDevices;
        this.applicationInfo = applicationInfo;
        this.placementRequests = pr;

        this.fogDeviceMap = new HashMap<>();
        for (FogDevice d : fogDevices) fogDeviceMap.put(d.getId(), d);

        // 筛选可部署节点 (Level <= 2)
        this.deployableNodes = new ArrayList<>();
        for (FogDevice dev : fogDevices) {
            if (dev.getLevel() <= 2) deployableNodes.add(dev);
        }
        this.deployableNodes.sort(Comparator.comparingInt(FogDevice::getId));

        // --- [新增] 调试打印：确认 RL 能看到哪些节点 ---
        System.out.println("\n=== RL Logic Initialized ===");
        System.out.println("Total FogDevices: " + fogDevices.size());
        System.out.println("Deployable Nodes (Candidate Actions): " + deployableNodes.size());
        for(FogDevice d : deployableNodes) {
            System.out.println(" - [" + d.getId() + "] " + d.getName() + " (Level: " + d.getLevel() + ")");
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

    private ActionResult executeAction(int actionNodeIndex) {
        if (currentModuleIndex >= placementQueue.size())
            return new ActionResult(null, 0, true);

        if (actionNodeIndex >= deployableNodes.size()) {
            return new ActionResult(buildStateRepresentation("Invalid Action"), -100.0, false);
        }

        QueuedModule curr = placementQueue.get(currentModuleIndex);
        FogDevice node = deployableNodes.get(actionNodeIndex);

        double currentMips = currentCpuLoad.getOrDefault(node.getId(), 0.0);
        int currentRam = currentRamLoad.getOrDefault(node.getId(), 0);

        boolean enough = (node.getHost().getTotalMips() - currentMips >= curr.moduleObj.getMips()) &&
                (node.getHost().getRam() - currentRam >= curr.moduleObj.getRam());

        double reward = 0.0;
        String desc;

        if (enough) {
            updateSimulatedLoad(node.getId(), curr.moduleObj);
            currentPlacementMap.put(curr.getKey(), node.getId());
            double loadBalanceBonus = (1.0 - (currentMips / node.getHost().getTotalMips())) * 5.0;
            reward = 10.0 + loadBalanceBonus;
            desc = "Placed " + curr.moduleName + " on " + node.getName();
        } else {
            reward = -100.0;
            desc = "Failed to place " + curr.moduleName + " on " + node.getName() + " (Full)";
        }

        currentModuleIndex++;
        boolean done = (currentModuleIndex >= placementQueue.size());
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

        for (int i = 0; i < MAX_NODES; i++) {
            if (i < deployableNodes.size()) {
                FogDevice dev = deployableNodes.get(i);
                double total = dev.getHost().getTotalMips();
                double used = currentCpuLoad.getOrDefault(dev.getId(), 0.0);
                state.add((total - used) / total);
                state.add(1.0);
                mask.add(true);
            } else {
                state.add(0.0); state.add(0.0);
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
            String appId = parts[0];
            String moduleName = parts[1];
            Application app = applicationInfo.get(appId);
            AppModule module = app.getModuleByName(moduleName);

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