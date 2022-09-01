set term png 
set out "scaling_theory.png"
set key t l
set logscale
ps = '1000 1500 2000 5000'
set key b l
set xlabel "P"
set ylabel "test error"
set title "theory 1hl mnist"
p "/storage/local/sebastianoariosto/rosiwork/deepRegression/theory_mnist_N1_500.txt" u 1:($2)/28.5 title  "theory N1 = 500" ,\
for [i=1:words(ps)] "/storage/local/sebastianoariosto/rosiwork/deepRegression/theory_mnist_N1_".word(ps,i).".txt" u 1:($2) title  "theory N1 = ".word(ps,i)
