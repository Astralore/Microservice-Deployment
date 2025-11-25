package org.fog.placement;

import org.fog.utils.Logger;

/**
 * Created by Samodha Pallewatta.
 */
public class PlacementLogicFactory {

    public static final int EDGEWART_MICROSERCVICES_PLACEMENT = 1;
    public static final int CLUSTERED_MICROSERVICES_PLACEMENT = 2;
    public static final int DISTRIBUTED_MICROSERVICES_PLACEMENT = 3;
    // ----------- 添加点 1: 定义 RL 算法标识 -----------
    public static final int RL_PLACEMENT = 4;
    // ----------- 添加结束 -----------

    // 缓存 RLPlacementLogic 实例，避免重复创建和启动 API 服务器
    private static RLPlacementLogic rlPlacementLogicInstance = null;
    private static final Object lock = new Object(); // 用于线程安全

    public MicroservicePlacementLogic getPlacementLogic(int logic, int fonId) {
        switch (logic) {
//            case EDGEWART_MICROSERCVICES_PLACEMENT:
//                return new EdgewardMicroservicePlacementLogic(fonId);
            case CLUSTERED_MICROSERVICES_PLACEMENT:
                return new ClusteredMicroservicePlacementLogic(fonId);
            case DISTRIBUTED_MICROSERVICES_PLACEMENT:
                return new DistributedMicroservicePlacementLogic(fonId);
            case RL_PLACEMENT:
                // 使用单例模式确保只有一个 RLPlacementLogic 实例和 API 服务器
                synchronized (lock) {
                    if (rlPlacementLogicInstance == null) {
                        rlPlacementLogicInstance = new RLPlacementLogic(fonId);
                    }
                    // 可以考虑更新 fonId 如果需要的话: rlPlacementLogicInstance.setFonId(fonId);
                }
                return rlPlacementLogicInstance;
            // ----------- 添加结束 -----------
            default: // 添加 default case
                Logger.error("Placement Logic Error", "Unknown placement logic type: " + logic);
                return null; // 或者抛出异常
        }

        // Logger.error("Placement Logic Error", "Error initializing placement logic for type: " + logic); // 这行现在是不可达的
        // return null;
    }

    // 可选：添加一个方法来显式停止 RLPlacementLogic 的服务器（例如在仿真结束后）
    public static void stopRLPlacementLogicServer() {
        synchronized (lock) {
            if (rlPlacementLogicInstance != null) {
                rlPlacementLogicInstance.postProcessing(); // 调用清理方法
                rlPlacementLogicInstance = null; // 清除实例引用
            }
        }
    }
}
