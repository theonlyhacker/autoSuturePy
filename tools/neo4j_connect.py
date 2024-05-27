from py2neo import Node, Relationship,Graph,NodeMatcher,Subgraph


test_graph = Graph(
    "http://localhost:7474",auth=("neo4j","12345678") ,name="neo4j"
)
# 删除所有：谨慎使用
test_graph.delete_all()
# 创建一个wound节点，包含子节点有伤口形状(包含子节点形状类别，特征区域)，大小，深度，伤口组织，伤口部位
wound_node = Node("Wound", name="wound")
shape_node = Node("Shape", type="形状类别", area="特征区域")
size_node = Node("Size", name="大小", unit="mm")
depth_node = Node("Depth", value="深度")
tissue_node = Node("Tissue", type="伤口组织")
position_node = Node("Position", location="伤口部位")

# 创建一个knot节点，包含子节点有缝合时长，缝合轨迹规划(这个包含子节点出入规划点，缝合深度，缝合形状)，缝合结果评价
knot_node = Node("Knot", name="方案")
duration_node = Node("Duration", value="缝合时长")
trajectory_node = Node("Trajectory", in_point="节点规划", depth="缝合深度", shape="缝合形状")
evaluation_node = Node("Evaluation", value="结果评价")

# 创建一个Devices节点，表示缝合设备的硬件信息以及实际的运动路线，包含子节点机械臂基本属性，缝合装置基本属性，相机基本属性，运动速率，待运动点集合
devices_node = Node("Devices", name="devices")
arm_node = Node("Arm", attributes="机械臂")
stitch_node = Node("Stitch", attributes="缝合机构")
camera_node = Node("Camera", attributes="相机")
sensor_node = Node("sensor", attributes="力传感器")
speed_node = Node("Speed", value="运动速率", unit="mm/s")
points_node = Node("Points", name="待运动点", unit="点数个数")

# 创建一个知识图谱节点，包含子节点有伤口，缝合，设备
knowledge_node = Node("Knowledge", name="knowledge")


# 创建一些关系，如wound拥有一个knot，knot拥有一个devices，wound和knot的子节点之间的关系等
has_knot = Relationship(wound_node, "HAS_KNOT", knot_node)
has_devices = Relationship(knot_node, "HAS_DEVICES", devices_node)
has_shape = Relationship(wound_node, "HAS_SHAPE", shape_node)
has_size = Relationship(wound_node, "HAS_SIZE", size_node)
has_depth = Relationship(wound_node, "HAS_DEPTH", depth_node)
has_tissue = Relationship(wound_node, "HAS_TISSUE", tissue_node)
has_position = Relationship(wound_node, "HAS_POSITION", position_node)
has_duration = Relationship(knot_node, "HAS_DURATION", duration_node)
has_trajectory = Relationship(knot_node, "HAS_TRAJECTORY", trajectory_node)
has_evaluation = Relationship(knot_node, "HAS_EVALUATION", evaluation_node)
has_arm = Relationship(devices_node, "HAS_ARM", arm_node)
has_stitch = Relationship(devices_node, "HAS_STITCH", stitch_node)
has_camera = Relationship(devices_node, "HAS_CAMERA", camera_node)
has_sensor = Relationship(devices_node, "HAS_SENSOR", sensor_node)

# add by lc

has_speed = Relationship(arm_node, "HAS_SPEED", speed_node)
has_points = Relationship(arm_node, "HAS_POINTS", points_node)
# has_sensor_stitch = Relationship(stitch_node, "HAS_SENSOR", sensor_node)
has_knoledge_wound = Relationship(knowledge_node, "HAS_WOUND", wound_node)
has_knoledge_knot = Relationship(knowledge_node, "HAS_KNOT", knot_node)
has_knoledge_devices = Relationship(knowledge_node, "HAS_DEVICES", devices_node)

# 创建一个子图，包含所有的节点和关系
subgraph = Subgraph([wound_node, shape_node, size_node, depth_node, tissue_node, position_node, 
                     knot_node, duration_node, trajectory_node, evaluation_node, devices_node, 
                     arm_node, stitch_node, camera_node,sensor_node, speed_node, points_node], 
                    [has_knot, has_devices, has_shape, has_size, has_depth, has_tissue, has_position, 
                     has_duration, has_trajectory, has_evaluation, has_arm, has_stitch, has_camera, 
                     has_speed, has_points,has_knoledge_devices,has_knoledge_knot,has_knoledge_wound,
                     has_sensor])

# 将子图添加到test_graph中
test_graph.create(subgraph)