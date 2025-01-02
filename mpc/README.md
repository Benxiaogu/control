`mpc_r0.py`是基于DR_CAN在bilibili提供的[Octave代码](https://www.bilibili.com/opus/666629407425691664)//www.bilibili.com/opus/666629407425691664使用python语言复现的，其中的实现原理也是按照DR_CAN在[视频](https://www.bilibili.com/video/BV1cL411n7KV?spm_id_from=333.788.videopod.sections&vd_source=9e51c239510424a92903fe86ed4d9e71)中在假设系统参考r=0的情况下进行推导得到的目标函数，如式(1)所示
$$
J=x_k^TM^T\bar{Q}Mx_k+2x_k^TM^T\bar{Q}CU_k+U_k^T(C^T\bar{Q}C+\bar{R})U_k \tag{1}
$$
上式的推导过程简化如下：
$$
J=\sum_k^{N-1} E_k^TQE_k + U_k^T\bar{R}U_k + E_N^TFE_N \tag{2}
$$
![](https://s2.loli.net/2025/01/03/zf7c5dhH2FyWuSr.png)



由于假设$r=0$，所以$e_k=y_k-r=x_k-r=x_k$，于是便得到了下式(具体推导过程在[视频](https://www.bilibili.com/video/BV1cL411n7KV?spm_id_from=333.788.videopod.sections&vd_source=9e51c239510424a92903fe86ed4d9e71)中有具体细节，此处省略)
$$
J=X_k^T\bar{Q}X_k + U_k^T\bar{R}U_k \tag{3}
$$
然后将式(4)代入上式，就会得到式(1)
$$
X_k=Mx_k+CU_k \tag{4}
$$
但是如果$r\neq0$，那么$e_k=x_k-r$，这里假设$r$是固定的，由于
$$
e_k = x_k-r \\
e_{k+1} = x_{k+1} - r \\
x_{k+1} = A*x_k + B*u_k \\
于是，e_{k+1}=A*e_k+B*u_k+(A-I)*r \\
(这里的*表示矩阵相乘)
$$
那么，就会得到系统误差矩阵，如下
$$
E_k=M*e_k+C*u_k+T*r \tag{5}
$$
这里的$M,C$与之前的相同，$T$矩阵如下
$$
T=
\begin{bmatrix}
0 \\
A-I \\
A*(A-I) \\
A^2*(A-I) \\
\vdots \\
A^{N-1}*(A-I)
\end{bmatrix} \tag{6}
$$
将代入到式(6)，得到目标函数如下
$$
J = U_k^T(C^T\bar{Q}C)U_k + 2*x_k^TM^T\bar{Q}CU_k+2*r^TT^T\bar{Q}CU_k+x_k^TM^T\bar{Q}Mx_k+2*x_k^TM^T\bar{Q}Tr+r^TT^T\bar{Q}Tr
$$
最终在`mpc.py`中实现

![](https://s2.loli.net/2025/01/03/jVkNxrAHvo9h7yS.png)

在程序`mpc.py`中设置的系统期望为$[10, 10]$，但是从图中可以看出并没有趋于10，推导过程应该是没有错的，目前推测是$r\neq0$时的其他假设或者前提存在问题，正在解决中...，如果您感兴趣的话希望可以在 issue 中交流一下

