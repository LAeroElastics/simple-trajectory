# simple-trajectory
OpenMDAOを利用して軌道最適化問題を解いてみました。連続時間の最適制御問題を最適化問題に変換してSQPで解いています。  
  技術的背景、実装などについては下記を参考にさせていただきました。  
  【Blog】軌道生成の基礎（Scipyのoptimizeを使った非線形最適制御問題の解法）: ina111様  
  http://qiita.com/ina111/items/fc7ae980c5568c0d0817  
  【論文】Direct Trajectory Optimization and Costate Estimation via an Orthogonal Collocation Method  
  http://vdol.mae.ufl.edu/JournalPublications/AIAA-20478.pdf   

  Solving Optimal Contorol Problem with OpenMDAO.   
  Converting continuous time optimal control problem into optimization problem,  
  solved using SQP method.  
  This example is trajectory optimization of flying object being dominated its   
  motion by Gravity and with some initial conditions.  
  Really simple EoM is implemented to evaluate usage of OpenMDAO on this type of problem.  

default condition is as follows:  
-Evaluation Function is arranged to minimizing time.  
  -State variables are velocity, position and path angle(NOT AOA).  
  -Can handle both equal and in-equal restraint condition.  

  These conditions are really easy to rearrange.  
