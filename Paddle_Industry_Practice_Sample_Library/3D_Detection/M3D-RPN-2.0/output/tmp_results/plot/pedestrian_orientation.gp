set term postscript eps enhanced color font "Helvetica" 20
set output "pedestrian_orientation.eps"
set size ratio 0.7
set xrange [0:1]
set yrange [0:1]
set xlabel "Recall"
set ylabel "Orientation Similarity"
set title "Pedestrian"
plot "pedestrian_orientation.txt" using 1:2 title 'Easy' with lines ls 1 lw 5,"pedestrian_orientation.txt" using 1:3 title 'Moderate' with lines ls 2 lw 5,"pedestrian_orientation.txt" using 1:4 title 'Hard' with lines ls 3 lw 5