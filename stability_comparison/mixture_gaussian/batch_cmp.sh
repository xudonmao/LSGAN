#!/bin/bash
export QT_QPA_PLATFORM=offscreen
for x in `seq 0 9`; do 
  python -u ls.py
  mkdir ls_$x
  mv *.png ls_$x

  python -u sig.py
  mkdir sig_$x
  mv *.png sig_$x
done

