set term png 
set out "scaling.png"
set key t l
set logscale
ps = '3000'
set multiplot layout 2,1
set title "train"
p for [i in ps] "run_P_".i."_replica_0.txt" u 1:2 w l title "P = ".i

set title "test"
p for [i in ps] "run_P_".i."_replica_0.txt" u 1:3 w l title "P = ".i