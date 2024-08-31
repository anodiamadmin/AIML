import csv, random, time

x_value = 0
total1 = 1000
total2 = 1000

fieldnames = ["x_value", "total1", "total2"]

with open('./data/autoUpdate.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

while True:
    with open('./data/autoUpdate.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writerow({"x_value": x_value, "total1": total1, "total2": total2})
        print(x_value, total1, total2)
        x_value += 1
        total1 += random.randint(-6, 8)
        total2 += random.randint(-5, 6)
    time.sleep(1)
