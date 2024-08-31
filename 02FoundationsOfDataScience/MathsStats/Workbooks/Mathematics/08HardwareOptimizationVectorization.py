import time
import numpy as np

ary_len = 10000000           # 10 million
my_ary1 = np.random.randint(0, 10, ary_len)
my_ary2 = np.random.randint(0, 10, ary_len)
sum_ary_c = np.zeros(ary_len)
sum_ary_v = np.zeros(ary_len)

# classic loop implementation
start_time = time.process_time() * 1000     # milli seconds
for i in range(ary_len):
    sum_ary_c[i] = my_ary1[i] + my_ary2[i]
end_time = time.process_time() * 1000
time_req_c = end_time - start_time

# vectorized implementation
start_time = time.process_time() * 1000
sum_ary_v = my_ary1 + my_ary2
end_time = time.process_time() * 1000
time_req_v = end_time - start_time

print(f'my_ary1 = {my_ary1}')
print(f'my_ary2 = {my_ary2}')
print(f'Sum array calculated by Classic Loop = {sum_ary_c}')
print(f'Time Required in Classic Loop = {time_req_c} milli seconds')
print(f'Sum array calculated by Vectorised Implementation = {sum_ary_v}')
print(f'Time Required in Vectorised Implementation = {time_req_v} milli seconds')
print(f'Difference in results = {np.sum(sum_ary_v-sum_ary_c)}')
print(f'While Vectorized Implementation takes {round(time_req_v/time_req_c*100, 3)} % of time')
