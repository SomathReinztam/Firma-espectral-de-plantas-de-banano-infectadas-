"""
L_out.py


"""

def L_out(L_in, padding, dilation, kernel_size, stride):
    x = L_in + 2*padding - dilation*(kernel_size-1) - 1
    y = x/stride
    z = y+1
    return int(z)

x = L_out(L_in=170, padding=4, dilation=1, kernel_size=8, stride=1)

print(x)
