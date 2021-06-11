import random

def _random_number(n):
    r = []
    a = 0
    while a < n:
        a += 1
        i = 0
        id_12 = ""
        id_10 = ""
        while True:
            i += 1
            if len(id_10) < 9:
                nb = random.randint(0, 9)
                id_10 += f"{nb}"
            if len(id_12) < 12:
                nb = random.randint(0, 9)
                id_12 += f"{nb}"
            if len(id_10) == 12:
                break
        r.append(id_10)
        r.append(id_12)
    return r

_list_r = _random_number(10000)
with open("/home/huyphuong/PycharmProjects/project_graduate/test/random_number_id.txt", "w") as fp:
    fp.write('\n'.join(_list_r))