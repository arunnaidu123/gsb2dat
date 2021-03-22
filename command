rm -rf B1937+21.250416.1280.dat file_l.dat
/home/arun/analysis/gsb2dat/gsb2dat -r1 ./node33R1/B1937+21.250416.1280.30ns.Pol-R1.dat -r2 ./node33R2/B1937+21.250416.1280.30ns.Pol-R2.dat -l1 ./node34L1/B1937+21.250416.1280.30ns.Pol-L1.dat -l2 ./node34L2/B1937+21.250416.1280.30ns.Pol-L2.dat -t ./node33R1/B1937+21.250416.1280.30ns.timestamp -S B1937+21 -n 512 -f 1292.0 -o B1937+21.250416.1280
dspsr B1937+21.250416.1280.dat  -cuda 0,1 -P polyco_new.dat -D 71.019727 -L 10 -b 256 -F 64:D -N B1937+21 -D 71.019727 -A -O B1937+21.250416.1280
pam -DFp -e DFp B1937+21.250416.1280.ar
psrplot -p time -D 'test.ps/ps' B1937+21.250416.1280.DFp
convert -density 300 -rotate 90 test.ps test.png
