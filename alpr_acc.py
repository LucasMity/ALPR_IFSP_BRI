def arqToDict(arquivo):
    Dict = {}
    with open(arquivo, 'r') as arq:
        linhas = arq.readlines()
        for linha in linhas:
             aux = linha.split(' ')
             aux[1] = aux[1].replace('\n','')
             try:
                Dict[aux[0]].append(aux[1])
             except:
                Dict[aux[0]] = []
                Dict[aux[0]].append(aux[1])
             aux.clear()
    return Dict

predict_file = './test_predicts/predicts/final_lp.txt'
real_file = './real/final.txt'

predict = arqToDict(predict_file)
real = arqToDict(real_file)

cont = 0
total = 0

for image in real:
    for i in real[image]:
        total += 1
        if image not in predict:
            continue
        for j in predict[image]:
            if i == j:
                print(i,j)
                cont += 1
                break

print(cont / total)
