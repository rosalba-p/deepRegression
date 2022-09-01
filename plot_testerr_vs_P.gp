set term png size 1000,1000
set out "scaling_mnist.png"
set logscale x
set key t r
set pointsize 1

ps = "100 400 1000"
pst = "500 1000"

point_types = "4 5 6"
point_colors = "red blue green"
point_colors2 = "blue green red" 

set multiplot layout 2,1
set ylabel "test error"
set xlabel "P"
p for [i=1:words(ps)] "scaling_lr_0.01_w_decay_1e-05_noise_0.0_bs_0_N_1000_N1_".word(ps,i).".txt" u ($1):($4) w p pt word(point_types,i) lt rgb word(point_colors,i) title "mnist N = 400 N1 = ".word(ps,i) ,\
for [i=1:words(pst)] "scaling_lr_0.01_w_decay_1e-05_noise_0.0_bs_0_N_196_N1_".word(pst,i).".txt" u ($1):($4) w p pt word(point_types,i) lt rgb word(point_colors2,i) title "mnist N = 196 N1 = ".word(pst,i) ,\
"/storage/local/sebastianoariosto/rosiwork/deepRegression/theory_mnist_N1_500.txt" u 1:($2)/28.5 title "theory N = 500"

set xlabel "P/N1"
p for [i=1:words(ps)] "scaling_lr_0.01_w_decay_1e-05_noise_0.0_bs_0_N_1000_N1_".word(ps,i).".txt" u ($1)/word(ps,i):($4) w p pt word(point_types,i) lt rgb word(point_colors,i) title "mnist N = 400 N1 = ".word(ps,i) ,\
for [i=1:words(pst)] "scaling_lr_0.01_w_decay_1e-05_noise_0.0_bs_0_N_196_N1_".word(pst,i).".txt" u ($1)/word(pst,i):($4) w p pt word(point_types,i) lt rgb word(point_colors2,i) title "mnist N = 196 N1 = ".word(pst,i) ,\
"/storage/local/sebastianoariosto/rosiwork/deepRegression/theory_mnist_N1_500.txt" u ($1)/word(pst,i):($2)/28.5 title "theory N = 500"
