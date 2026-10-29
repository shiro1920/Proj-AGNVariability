## add new color table 

def RGB2Hex(rgb_list):
    
    color_list = {}
    
    for k in range(len(rgb_list)):
        RGB = rgb_list[k].split(',')           
        color = '#'
        for i in RGB:
            num = int(i)               
            color += str(hex(num))[-2:].replace('x', '0').upper()
            
        key = "C"+str(k)
        color_list[key] = str(color)
    
    return color_list