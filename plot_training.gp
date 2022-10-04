set term png size 800,800
set out "dynamics.png"
set key t l
set key font ",5"
set logscale
ps = '69 110 176 282 452 723 1157 1852 2963'
ps = '2963'
replicas = '0 1 2 3 4 5 6 7 8 9'
set multiplot layout 2,1
set title "train"
p for [i in ps] for [j in replicas] "run_P_".i."_replica_".j.".txt" u 1:2 w l title "P = ".i." run = ".j

set title "test"
p for [i in ps] for [j in replicas] "run_P_".i."_replica_".j.".txt" u 1:3 w l title "P = ".i." run = ".j