使用Gasualdo Scutari Francisco Facchinei《Real and Complex Monotone Communication Games》的算法(即N+1player)，计算出了变分均衡的值。

源代码如链接🔗所示：

https://github.com/MrDotJ/repetition-paper/tree/master/source/variational_equilibrium

测试系统及结果📊：(绿线表明收敛曲线)

1. 简单的2节点系统

   ```
   player1_info = {
       'index': 0,
       'demand_ref': 15,  # 基准负荷
       'supply_max': 5,   # 最大生产功率
       'demand_max': 25,  #最大负荷  下同
   }
   player2_info = {
       'index': 1,
       'demand_ref': 20,
       'supply_max': 45,
       'demand_max': 30,
   }
   ```

   系统的收敛曲线如图所示：

   ![image](results/variational_equilibrium/2-node-elec.svg)

2. 三节点系统

   其系统全连接，具体配置如下

   ```
   player1_info = {
       'index': 0,
       'demand_ref': 20,
       'supply_max': 10,
       'demand_max': 30,
   }
   player2_info = {
       'index': 1,
       'demand_ref': 20,
       'supply_max': 30,
       'demand_max': 25,
   }
   player3_info = {
       'index': 2,
       'demand_ref': 20,
       'supply_max': 30,
       'demand_max': 25,
   }
   ```

   该系统三节点互联，系统得到的收敛曲线如图

   ![image](results/variational_equilibrium/3-node-elec.svg)

   其中2，3节点配置完全相同，所以曲线有重合☝

   改变不同节点的价格成本情况，

   ```
   player2_info = {
       'supply_a': 2.5,
       'supply_b': 0.1,
       'demand_a': 10,
   }
   player3_info = {
       supply_a': 3,
       'supply_b': 0.1,
       'demand_a': 10,
   }
   ```

   得到的新的均衡如下：

   ![image](results/variational_equilibrium/3-node-elec-diff-config.svg)

3. 五节点系统

   五节点系统拓扑如图：

   ```
   #    0     1      2
   #    o-----o------o
   #          |      |
   #          o      o
   #          3      4
   ```

   系统配置如源代码所示📜

   计算结果如图(由于结果较多，图片仅展示1<-->0, 1<-->2, 1<-->3的结果):

   ![image](results/variational_equilibrium/5-node-elec.svg)

   

4. 综合能源(气/热/无储能)五节点系统

   主要参考了老师的文章《A Generalized Nash Equilibrium Approach for Autonomous Energy Management of Residential Energy Hubs》👈对于Energy Hub的建模，其中包括gas--gas turbine, gas--gas furnace, 但是忽略了与公网的连接，仅保留区域间的互联，并且忽略了所有的储能装置

   拓扑采用的是相同的五节点

   系统的配置如源代码所示

   计算结果如图(由于结果较多，图片仅展示1<-->0, 1<-->2, 1<-->3的结果):

   ![image](results/variational_equilibrium/5-node-IES.svg)

5. 一个有趣的现象是，对于情况2(三节点)中最终结果，施加函数：

   Price = Constant - Coefficient * (Power_real - Power_reference)， 即

   ```
   new_price = old_price - 0.1 * (demand_value - self.demand_ref)
                                #    优化结果         参考值
   ```

   发现结果基本相同

   ```
   #[[[0, 5.3261481100402515, 5.3261481100402515],
   #  [5.325721551707392, 0, 5.325721551707392],
   #  [5.325721551707392, 5.325721551707392, 0]],
   ```

    感觉可能是配置相同导致的，也可能是隐含了变分均衡的物理意义



