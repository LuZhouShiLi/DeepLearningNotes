# pytorch合并拆分运算

## 简述


* 如果维度少了直接添加一个维度
* 如果每一个维度的偏少，直接扩展每一个维度的元素数量


![图 1](../images/44b10bb79f6af97f44931f95922431aa90c61b1ee2240fa9c155db66da551ae3.png)  


## Cat合并操作

![图 2](../images/a2a4539aec161ed78a73f672eb4ef988032f577dfcdc35486b2c2aa334a68ae0.png)  

![图 3](../images/3a2f7199060e115e34490f8273e73600ebb307c7932fc0fb02596867cbd29cf5.png)  


**cat操作的维度必须是一致的，只能有一个维度不一样（拼接的维度），但是其他的维度都必须是一样的**


**案例：为一张RGB照片再添加一个通道变成四通道**

![图 4](../images/1291c2fa7790c1978bedac952b7042bba6b9c24b3b41164cbe00416f8156bdbb.png)  

## stack-插入新的维度操作

**stack操作两个张量的维度必须全部保持一致**

![图 6](../images/b0ffc34818fbe317dca1965dd705505ac465beedd6bdee3a12e2de4b2bdd11b4.png)  



**stack操作是创建一个新的维度**


## splict拆分操作

**第一个参数是拆分的尺寸，第二个参数是拆分的维度**

![图 7](../images/d35864bedcad655d38cdcbb0442dd6f5363a7f274a6dee972af9bac2307f7712.png)  


