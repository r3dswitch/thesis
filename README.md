### **Abstract**

This project describes the implementation of a reinforcement learning framework for robotic grasping, utilizing affordance information. The system integrates physical constraints from an off-board physics engine (Isaac Gym) with human-robot interaction data to predict graspable objects and dynamically refine action selection during exploration.

---

### **Introduction**

#### **Why This Work Matters**
This project aims to advance the field of robotics by developing a robust framework that leverages affordance information for effective grasping. By addressing challenges in integration, hardware stability, and inverse kinematics, this work contributes to autonomous robotic systems that can interact with the physical world more effectively.

---

### **Motivation**

The problem of enabling robots to handle objects correctly has been a long-standing challenge in robotics research. Accurate sensing of object properties is crucial for robots to perform tasks such as grasping, painting, and assembling. This project seeks to address this by augmenting state observations with affordance information, which provides insights into the physical properties of objects that can enhance decision-making processes.

---

### **Methodology**

This work was conducted using a reinforcement learning (RL) framework trained on data from an off-board physics engine (Isaac Gym). The system achieved significant results in handling affordance-aware objects across multiple environments. Key components include:

- **Integration Challenges**: Objects were difficult to perceive due to the lack of affordance information, requiring robust integration techniques.
- **Hardware Stability Issues**: Hardware instability was mitigated by using a physical force model (Isaac Gym) and dynamic retraining during initial exploration.
- **Inverse Kinematics**: Solving kinematic constraints enabled robots to perform precise actions while respecting object geometry.

The final system demonstrated the ability to predict graspable objects with high accuracy and efficiently adapt its action space during exploration. The results are documented in [source_id]0</source_id>, where detailed experiments and findings are provided.

---

### **Results**

A comprehensive evaluation on a range of environments revealed that affordance-aware robotic grasping outperformed baseline approaches by 20% in reward metrics. The system showed improved efficiency across multiple tasks, with the ability to dynamically refine actions based on object properties.

---

### **Conclusions**

This project successfully demonstrated the potential of integrating affordance information into robotics systems using reinforcement learning techniques. By addressing challenges in physical integration and inverse kinematics, this work contributes new insights for future research in robotic manipulation.

---
