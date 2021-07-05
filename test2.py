import numpy as np
from bitstring import BitArray
pwd = "/Users/bytedance/Desktop/MacroModel/cim_test_adc_out/"
def hex2bin(input, width=32):
  return BitArray("0x"+input, length=width).bin


list = [[] for i in range(16)]
for i in range(16):
    pwd2 = pwd + "CIM_" + str(i) + "_adc_hex.txt"
    fo = open(pwd2, "r")
    lines = fo.readlines()
    for line in lines:
        b = hex2bin(line, 4)
        list[i].append(b)


ff = open(pwd + "out.txt", "w")
out = ""
for k in range(64):
    for j in range(16):
        out = list[j][k] + out
    ff.write(out)
    ff.write("\n")
    out = ""