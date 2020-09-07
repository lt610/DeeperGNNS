
def generate_shells():
    with open('../shells/tmp.txt', 'a') as f:
        line = 'python train_vsgc.py --dataset ogbn-arxiv --num_layers {} --alpha 0.5\n'
        for i in range(2, 9):
            command = line.format(i)
            f.write(command)


if __name__ == '__main__':
    generate_shells()