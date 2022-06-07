dir_name = 'configs/infonerf/synthetic'
for name  in ['drums', 'chair', 'ficus', 'hotdog', 'materials', 'lego', 'mic', 'ship']:
    with open(f'{dir_name}/lego.txt', 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('lego',f'{name}')
    with open(f'{dir_name}/{name}.txt','w') as file:
        file.write(filedata)
