---
typora-copy-images-to: results\peer-to-peer energy sharing among smart grid
---

# repetition paper

there are four buildings as follow

`svac : adjustable HVAC units (index 0) with energy storage system`

`sea_1 sea_2 : shiftable electrical appliances (index 1, index 2) without energy storage system`

`fcs : flexible commerical services (index 3) without ess`

detail information is shown as follow: 

```
svac_info_1 = {
    'build_index': 0,
    'T_outside': [26, 26, 28, 29, 30, 31, 32, 32, 33],
    'C': 3.3,
    'R': 1.35,
    'M': 50,
    'alpha': 1,
    'nu': 0.185,
    'grid_buy_price': 0.25,
    'grid_sell_price': 0.1,
    'connection_n': 4,
    'exchange_power_limit': 100,
    'cycle': 9,
    'generator' : [3.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.5, 3.0]
}
ess_info_1 = {
    'ess_enable': 1,
    'charge_coefficient': 0.94,
    'charge_power_limit': 80,
    'charge_cost': 0.01,
    'discharge_coefficient': 1.06,
    'discharge_power_limit': 80,
    'discharge_cost': 0.01,
    'storage_capacity': 340,
    'storage_initial': 0.17647,
    'SoC': 0.88235
}
ess_info_2 = {
    'ess_enable': 0,
    'charge_coefficient': 0.94,
    'charge_power_limit': 80,
    'charge_cost': 0.01,
    'discharge_coefficient': 1.06,
    'discharge_power_limit': 80,
    'discharge_cost': 0.01,
    'storage_capacity': 340,
    'storage_initial': 0.17647,
    'SoC': 0.88235
}
sea_info_1 = {
    'build_index': 1,
    'load_min': 10,
    'load_max': 40,
    'load_ref': [20, 30, 20, 30, 20, 30, 10, 20, 20],
    'grid_buy_price': 0.25,
    'grid_sell_price': 0.1,
    'M': 50,
    'D': 200,
    'beta': 0.1,
    'exchange_power_limit': 100,
    'cycle': 9,
    'generator' : [4.2, 6.0, 7.0, 7.5, 7.8, 7.5, 6.0, 4.0, 2.0],
    'connection_n' : 4
}
sea_info_2 = {
    'build_index': 2,
    'load_min': 10,
    'load_max': 40,
    'load_ref': [20, 40, 30, 40, 30, 30, 10, 20, 20],
    'grid_buy_price': 0.25,
    'grid_sell_price': 0.1,
    'M': 50,
    'D': 240,
    'beta': 0.1,
    'exchange_power_limit': 100,
    'cycle': 9,
    'connection_n' : 4,
    'generator': [5.2, 7.0, 8.3, 9.0, 9.2, 8.2, 7.6, 5.4, 3.0]
}
fcs_info_3 = {
    'build_index': 3,
    'lam': 3,
    'load_min': 15,
    'load_max': 35,
    'grid_buy_price': 0.25,
    'grid_sell_price': 0.1,
    'connection_n': 4,
    'cycle': 9,
    'exchange_power_limit': 100,
    'generator': [60, 80, 100 , 110, 112, 100, 90, 60, 40]
}
```

***

The problem is optimized through ADMM described in paper 'Peer-to-Peer ..' page 5

the optimized result is shown as follow :

![build0](C:\Users\Mr.J\Documents\GitHub\repetition-paper\results\peer-to-peer energy sharing among smart grid\build0.svg)

![build1](C:\Users\Mr.J\Documents\GitHub\repetition-paper\results\peer-to-peer energy sharing among smart grid\build1.svg)

![build2](C:\Users\Mr.J\Documents\GitHub\repetition-paper\results\peer-to-peer energy sharing among smart grid\build2.svg)

![build3](C:\Users\Mr.J\Documents\GitHub\repetition-paper\results\peer-to-peer energy sharing among smart grid\build3.svg)

![differencebuild](C:\Users\Mr.J\Documents\GitHub\repetition-paper\results\peer-to-peer energy sharing among smart grid\differencebuild.svg)

![build0](F:\untitled\build0.svg)



the first three pictures show the optimized results for each building

​	the solid lines show the optimized load

​	the three dashed lines show the exchange power with all other buildings, we can see build_3 shared the most energy with other buildings because its generator has the biggest capacity which can be seen in the configuration above

the last picture shows that the algorithm converges, the f-norm of the difference between each iterator decreases quickly to zero(almost) 