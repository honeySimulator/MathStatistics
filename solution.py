from numpy import *
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv(r'/Users/Helga/Downloads/Telegram Desktop/Marshalikin_data.csv')
df = pd.DataFrame(data, columns=['gender', 'math.score', 'reading.score', 'writing.score'])
gender = df['gender']
math_score = df['math.score']
reading_score = df['reading.score']
writing_score = df['writing.score']

# parsing
marks = empty((3, 1000), int)
balls = array((math_score, reading_score, writing_score))

marks_math_female = zeros(4, float)
marks_math_male = zeros(4, float)
math_k = zeros(4, float)
reading_k = zeros(4, float)
writing_k = zeros(4, float)

count_female = 0
count_male = 0
balls_math_female = []
balls_math_male = []

for i in range(3):
    for j in range(1000):
        if balls[i][j] < 60:
            marks[i][j] = 2
        if 60 <= balls[i][j] < 74:
            marks[i][j] = 3
        if 74 <= balls[i][j] < 90:
            marks[i][j] = 4
        if balls[i][j] >= 90:
            marks[i][j] = 5
        if i == 0:
            if gender[j] == 'female':
                balls_math_female.append(balls[0][j])
                count_female += 1
            if gender[j] == 'male':
                balls_math_male.append(balls[0][j])
                count_male += 1

for i in range(1000):
    for j in range(2, 6):
        if marks[0][i] == j:
            math_k[j - 2] += 1
            if gender[i] == 'female':
                marks_math_female[j - 2] += 1
            if gender[i] == 'male':
                marks_math_male[j - 2] += 1
        if marks[1][i] == j:
            reading_k[j - 2] += 1
        if marks[2][i] == j:
            writing_k[j - 2] += 1

# 1
n = 1000
math_mean = sum(math_score) / n
print('1.')
print('Mean: ', math_mean)

math_variance = 0
for i in range(n):
    math_variance += pow(math_score[i], 2) - pow(math_mean, 2)
math_variance = math_variance/n
print('Variance: ', math_variance)

math_med = sort(math_score)
math_med = (math_med[499] + math_med[500]) / 2
print('Median: ', math_med)

# confidence intervals
s = sqrt(n/(n-1)*math_variance)
t = 1.646

left = math_mean - t/sqrt(n)*s
right = math_mean + t/sqrt(n)*s
print('Mean', left, ';', right)


q1 = 1074.679
q2 = 927.594

left = (n * pow(s, 2) / q1)
right = (n * pow(s, 2) / q2)
print('Variance', left, ';', right, '\n')

plt.hist(math_score)
plt.title('histogram')
plt.xlabel('distribution')
plt.ylabel('frequency')
plt.show()

pd.DataFrame(math_score).boxplot()
plt.title('box-plot')
plt.xlabel('distribution')
plt.ylabel('count')
plt.show()

# female
fem_mean = sum(balls_math_female) / count_female
print('Mean female: ', fem_mean)

fem_var = 0
for i in range(count_female):
    fem_var += pow(balls_math_female[i], 2) - pow(fem_mean, 2)
fem_var = fem_var/count_female
print('Variance female: ', fem_var)

balls_math_female = sort(balls_math_female)
print('Median female: ', (balls_math_female[246] + balls_math_female[247]) / 2)

# male
male_mean = sum(balls_math_male) / count_male
print('Mean male: ', male_mean)

male_var = 0
for i in range(count_male):
    male_var += pow(balls_math_male[i], 2) - pow(male_mean, 2)
male_var = male_var/count_male
print('Variance male: ', male_var)

balls_math_male = sort(balls_math_male)
print('Median male: ', (balls_math_male[252] + balls_math_male[253]) / 2, '\n')

# confidence intervals
# female

s = sqrt(count_female/(count_female-1)*fem_var)
t = 1.647

left = fem_mean - t/sqrt(count_female)*s
right = fem_mean + t/sqrt(count_female)*s
print('Mean female', left, ';', right)

q1 = 546.813
q2 = 443.459

left = (count_female * pow(s, 2) / q1)
right = (count_female * pow(s, 2) / q2)
print('Variance female', left, ';', right)

# male
s = sqrt(count_male/(count_male-1) * male_var)
t = 1.647

left = male_mean - t/sqrt(count_male)*s
right = male_mean + t/sqrt(count_male)*s
print('Mean male', left, ';', right)

q1 = 559.438
q2 = 454.835

left = (count_male * pow(s, 2) / q1)
right = (count_male * pow(s, 2) / q2)
print('Variance male', left, ';', right, '\n')

# female
plt.hist(balls_math_female)
plt.title('female')
plt.xlabel('distribution')
plt.ylabel('frequency')
plt.show()

pd.DataFrame(balls_math_female).boxplot()
plt.title('female')
plt.xlabel('distribution')
plt.ylabel('count')
plt.show()

# male
plt.hist(balls_math_male)
plt.title('male')
plt.xlabel('distribution')
plt.ylabel('frequency')
plt.show()

pd.DataFrame(balls_math_male).boxplot()
plt.title('male')
plt.xlabel('distribution')
plt.ylabel('count')
plt.show()

# 2
arr = zeros((2, 4), int)
for j in range(4):
    arr[0, j] += marks_math_female[j]
    arr[1, j] += marks_math_male[j]

vi_ = list(map(sum, arr))
vj_ = list(map(sum, zip(*arr)))

pk_ = zeros((2, 4), float)
for j in range(2):
    for i in range(4):
        pk_[j, i] = (vi_[j] * vj_[i]) / n
i_freq = [a/n for a in vi_]
j_freq = [a/n for a in vj_]

Chi_square = 0
for i in range(2):
    for j in range(4):
        Chi_square += pow(arr[i, j] - n * i_freq[i] * j_freq[j], 2) / (n * i_freq[i] * j_freq[j])

print('\n2. Chi_square =', Chi_square)
if Chi_square > 7.815:
    print("Marks depend on the gender")
else:
    print("Marks DON'T depend on the gender")

print('table:')
print(arr)
print('i-sums:', vi_)
print('j-sums:', vj_)
print('theoretic i-frequency : ', i_freq)
print('theoretic j-frequency : ', j_freq)
print('theoretic values: ')
print(pk_, '\n')

# 3
pk = list(map(sum, zip(*array((math_k, reading_k, writing_k)))))
pm = list(map(sum, array((math_k, reading_k, writing_k))))

pk_ = zeros(4, float)
for i in range(4):
    pk_[i] = pk[i] / sum(pm)

Chi_square = 0
chi_1 = chi_2 = chi_3 = 0
for j in range(4):
    chi_1 += 1 / 1000 / pk[j] * pow(math_k[j] - n * pk[j] / 3000, 2)
    chi_2 += 1 / 1000 / pk[j] * pow(reading_k[j] - n * pk[j] / 3000, 2)
    chi_3 += 1 / 1000 / pk[j] * pow(writing_k[j] - n * pk[j] / 3000, 2)
Chi_square = 3000*(chi_1 + chi_2 + chi_3)

print('3. Chi_square =', Chi_square)
if Chi_square > 12.592:
    print("samples are not uniform")
else:
    print("samples are uniform")
print('table:')
print(math_k)
print(reading_k)
print(writing_k)
print('i-sums:', pm)
print('j-sums:', pk)
print('frequency: ', pk_)
print('theoretic values: ', pk_*n)

if __name__ == '__main__':
    print('\n')
