# In the name of God
import random
#print('The analysis library has been ran')
import pandas as pd
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
import sys
def p(*args, sep=' ', end='\n', file=sys.stdout, flush=False):

    # Join the arguments with the specified separator
    output = sep.join(map(str, args))
    # Write the output to the specified file or stdout
    file.write(output + end)
    # Flush the output if required
    if flush:
        file.flush()

# A Function to load data
def loadData(path):
    df = pd.read_csv(path)
    return df


def loadExcel(path):
    try:
        data = pd.read_excel(path)
        print('The data has been successfully read')
    except FileNotFoundError:
        print(f'Error : the file {path} has not been found')
    return data

def allColumns(permision):
    '''A function to set setting in order to see all the columns'''
    if permision == True:
        pd.set_option("display.max_columns", None)
        pd.set_option('display.width', 1000)


def saveData(data, path):
    try:
        data.to_csv(path, index=False)
        print("DataFrame saved ")
    except Exception as e:
        print("Error:", e)


def renameColumns(data, main, substitude):
    data.rename(columns={main: substitude}, inplace=True)

def change_column_data_Type(serie,new_data_type : str,print_previous_data_type=False):
    if print_previous_data_type :
        p(str(serie.dtype))
    else :
        pass
    serie = serie.astype(new_data_type)
    return serie

def dataCleaning(data):
    data_clean = data.drop_duplicates()
    data_clean = data_clean.dropna()
    return data_clean


def frequencyInquire(data, columnName):
    return data.groupby(columnName).size()


def valueCount(data, columnName):
    return data[columnName].value_counts().sort_values(ascending=False).to_string()


def outLineCleaner(data, column):
    Q1 = data[column].quantile(0.25)
    Q2 = data[column].quantile(0.5)
    Q3 = data[column].quantile(0.75)
    print(f'Your data quantile is:\n Q1:{Q1} \n Q2:{Q2} \n Q3:{Q3}\n#### ')
    IQR = Q3 - Q1
    lowerBound = Q1 - 1.5 * IQR
    upperBound = Q3 + 1.5 * IQR
    return data[(data[column] >= lowerBound) & (data[column] <= upperBound)]


def correlationRate(column1, column2):
    return column1.corr(column2)


def allCorellations(data, dependentVar):
    cor = {}
    datac = list(data.columns)
    datac.remove(dependentVar)
    for i in range(0, len(datac)):
        cor.update({datac[i]: correlationRate(data[datac[i]], data[dependentVar])})
        sortedCore = dict(sorted(cor.items(), key=lambda item: abs(item[1]), reverse=True))
    return (sortedCore)


def returnColumnname(data):
    return list(data.columns)


def reordering(data, dependentVar):
    orderedColumn = list(allCorellations(data, dependentVar).keys())
    reorderedData = data[orderedColumn]
    reorderedData = pd.concat([reorderedData, data[dependentVar]], axis=1)
    return reorderedData



def randomDataFrame(columnsNumber,rowsNumber,decimal_number=2):
    data = {}
    column_chracter_askii = 65
    for var in range(columnsNumber):

        lower_bound = random.randint(1,20000)

        length = random.randint(1,20000)

        data.update({
        f'{chr(column_chracter_askii)}':[round(random.uniform(lower_bound,lower_bound+length),decimal_number) for t in range(rowsNumber)]
                    })
        column_chracter_askii +=1
    saveData(pd.DataFrame(data),'randomData.csv')
    print('The proccess has finished and a random data has been saved to \'randomData.csv\'')


def rowDeleter(data, Index, Inplace=False):
    new_data = data.drop(index=Index, inplace=Inplace)
    return new_data


def ExcelSaver(data, addres, Index=False):
    data.to_excel(addres, index=Index, engine='openpyxl')
    print('Data has been successfully saved')


def writeFile(string, nameOfFile):
    f = open(f'{nameOfFile}.txt', 'a')
    f.write(string)
    f.close()
    return 0


def findPackageAdress(packName):
    print('we are in the fuction body')
    import pkgutil
    packInfo = pkgutil.find_loader(packName)
    if packInfo is not None:
        packPath = packInfo.path
        print(f'The package path is\'{packPath}\' ')
    else:
        print(f'the {packName} has not been found')
    return packPath

def get_function_documentation(function):
    '''
    :param function: the name of func you want. The library must be mentioned. for eg : pandas.read_csv
    :return: It returns 0 and documentation printed automaticly
    '''
    from inspect import getsource
    p(getsource(function))


def FindIndexInRang(LowerBound, UpperBound, data, CO=True, CC=False, OO=False, OC=False):
    b = []
    if CC:
        for index, value in enumerate(data):
            if value >= LowerBound and value <= UpperBound:
                b.append(index)
    elif OO:
        for index, value in enumerate(data):
            if value > LowerBound and value < UpperBound:
                b.append(index)
    elif OC:
        for index, value in enumerate(data):
            if value > LowerBound and value <= UpperBound:
                b.append(index)
    elif CO:
        for index, value in enumerate(data):
            if value >= LowerBound and value < UpperBound:
                b.append(index)
    return b

def ranged_data(LowerBound,UpperBound,data,Column,CO=True, CC=False, OO=False, OC=False):
    if CC and UpperBound!= 'f' and LowerBound != 'f' :
        # print('1')
        return data.query(f"`{Column}`>={LowerBound} & `{Column}`<={UpperBound}")
    elif OO and UpperBound!= 'f' and LowerBound != 'f':
        # print('2')
        return data.query(f"`{Column}`>{LowerBound} & `{Column}`<{UpperBound}")
    elif OO and UpperBound== 'f' and LowerBound != 'f' :
        # print('6')
        return data.query(f"`{Column}`>{LowerBound}")
    elif OC and UpperBound!= 'f' and LowerBound != 'f':
        # print('3')
        return data.query(f"`{Column}`>{LowerBound} & `{Column}`<={UpperBound}")
    elif OO and UpperBound!= 'f' and LowerBound == 'f' :
        # print('8')
        return data.query(f"`{Column}`<{UpperBound}")
    elif CO and UpperBound!= 'f' and LowerBound != 'f' :
        # print('4')
        return data.query(f"`{Column}`>={LowerBound} & `{Column}`<{UpperBound}")
    elif CO and LowerBound == 'f' and UpperBound != 'f' :
        # print('7')
        return data.query(f"`{Column}`<={UpperBound}")
    elif CO and UpperBound == 'f' and LowerBound != 'f' :
        # print('5')
        return data.query(f"`{Column}`>={LowerBound}")
    else :
        print('An error acured in ranged_data method')

def convert_to_numeric(data):
    '''Use this method when you get an expected dtype as object'''
    return data.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'object' else x)

def columns_name_fix(data):
    columnsName = data.columns
    modifyingColumnsName = {}
    for item in columnsName :
        striped_item = item.strip()
        modify_item = striped_item.translate(str.maketrans(' .','__'))
        modifyingColumnsName.update({item :modify_item})
    return data.rename(columns=modifyingColumnsName)

def frequency_by_devision(series,devision_number):
    max_input = series.max()
    min_input = series.min()
    length_of_range = (max_input - min_input)/devision_number
    bounderies = [min_input]
    for var in range(devision_number-1):
        bounderies.append(bounderies[-1]+length_of_range)
    bounderies.append(max_input)
    sizes = []
    for var in range(len(bounderies)-1):
        sizes.append(len(pd.DataFrame(list(series),columns=['value']).query(f"{bounderies[var]}< value <= {bounderies[var+1]}")))
    sizes[0] = sizes[0] + len(pd.DataFrame(list(series),columns=['value']).query(f"{bounderies[0]} == value"))
    return sizes

def drop_columns(data,*cloumns):
    for item in cloumns :
        data = data.drop([item],axis=1)
    return data

def data_preproccessing(data,dependent_var,test_size=0.2,random_state=2):
    from sklearn.model_selection import train_test_split
    data = columns_name_fix(data)
    data = reordering(data,dependent_var)
    X = data.drop([dependent_var], axis=1)
    y = data[dependent_var]
    spilited_X_y = train_test_split(X,y,test_size=test_size,random_state=random_state)
    return spilited_X_y

def normal_column_score(serie,print_permission=False):
    '''
    :param serie: provide a series or single data column
    :return: score which indicates similarity to normal distribution.
    '''
    from scipy.stats import norm
    from scipy.stats import entropy
    def js(p, q):
        p1 = p / np.linalg.norm(p, ord=1)
        q1 = q / np.linalg.norm(q, ord=1)
        m = 0.5 * (p1 + q1)
        r = 0.5 * (entropy(p1, m) + entropy(q1, m))
        # p(f'p1 is :{p1}\nq1 is {q1}\n m is {m}\nr is {r} ')
        return r

    ideal = norm.rvs(size=len(serie), loc=serie.mean(), scale=serie.std())
    icount, idevision = np.histogram(ideal, bins=100)
    count, devision = np.histogram(serie, bins=100)
    p(f'Your column has js score of : {js(count,icount)}')
    if print_permission :
        print('The js score for provided normal distribution is:',js(count,icount))
    else :
        p('Print permission failed')
    return js(count,icount)


from random import choice
class RandomWalk():

    '''A class to generate random walks.'''
    def __init__(self,num_points=5000):
        self.num_points = num_points
        '''All walks start at 0,0'''
        self.x_values = [0]
        self.y_values = [0]
    def fill_walk(self):

        while len(self.x_values) < self.num_points :
            x_direction = choice([1,-1])
            x_distance = choice([0,1,2,3,4])
            x_step = x_direction * x_distance
            y_direction = choice([1, -1])
            y_distance = choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance
            if x_step ==0 and y_step == 0 :
                continue
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step
            self.x_values.append(next_x)
            self.y_values.append(next_y)

class DATA():
    def __init__(self,adress):
        self.adress = adress
        self.dataType = self.adress.split('.')[-1]
        if self.dataType.lower() == 'csv':
            self.data = loadData(self.adress)
        if self.dataType.lower() == 'xlsx' :
            self.data = loadExcel(self.adress)
        self.columnS =  self.data.columns
        self.arrayData = self.data.to_numpy()

    def __str__(self):
        return f'Your data has {self.data.shape[0]} rows and {self.data.shape[-1]} columns'
def uniform_distribution(size,loc,scale,bins='f',kde='f',print_permission=False):
    '''
    :param size: the size of numbers that should be produce
    :param loc: minimum data
    :param scale: maximum data
    :param bins: optional.The amount of bin if ploting is needed
    :param kde: optional. If distribution probibality is needed to be ploted
    :param print_permission: If it is needed to print array in sys.out
    :return: data array from numpy
    '''
    from scipy import stats
    data_uniform = stats.uniform.rvs(size=size, loc=loc, scale=scale)
    if bins != 'f' and kde != 'f' :
        from seaborn import displot
        from matplotlib.pyplot import show
        displot(data_uniform,bins=bins,kde=kde)
        show()
    else :
        pass
    if print_permission :
        print('final',np.array(data_uniform))
    else :
        pass

    return data_uniform

def normal_distribution(size,loc,scale,bins='f',kde='f',print_permission=False):
    '''
    :param size: The lenghth of the needed array.
    :param loc: mean of data.
    :param scale: standard variation of data
    :param bins: optional.count of devision of range data
    :param kde: optional.bolean. If you want plot distribution probability.
    :param print_permission: If it is needed to print array in sys.out
    :return: data array from numpy
    '''
    # p('Inside normal distribution function body')
    from scipy.stats import norm
    data_normal = norm.rvs(size=size,loc=loc,scale=scale)
    if print_permission :
        p(data_normal)
    else :
        pass

    if bins !='f' and kde !='f' :
        from seaborn import displot
        from matplotlib.pyplot import show
        displot(data_normal,bins=bins,kde=kde)
        show()
    else :
        pass

    return data_normal

def bernoulli_distrbution(size,chance_of_success,print_permission=False,return_value_count = False):
    '''
    :param size: Trial counts
    :param chance_of_success: p or chance of success. between 0 & 1.
    :param print_permission: print the array that is combination of 0 & 1.
    :param return_value_count: it will count number of 0 and number 1 occurance.
    :return: array that with size of number of trial that is combination of 0 & 1.
    '''
    from scipy.stats import bernoulli
    data_bernoulli = bernoulli.rvs(size=size,p=chance_of_success)
    if print_permission :
        p(data_bernoulli)
    else :
        p('bernoulli data print proccess has failed!')
    if return_value_count :
        Counts = np.unique_counts(data_bernoulli)
        dict = {}
        for var in range(0,1+1):

            dict.update({str(Counts.values[var]):int(Counts.counts[var])})
        p(dict)
    else :
        p('return_value_count has failed!')

def binom_distribution(size_of_array,number_of_trial,chance_of_success,bins='f',kde='f',print_permission=False):
    '''
    :param size_of_array:  length of produced array.
    :param number_of_trial: In each element of array how much trial musst be conducted.
    :param chance_of_success: between 0 & 1
    :param bins: number of devision on range of data
    :param kde: True or False. plot kde or not.
    :param print_permission: print produced data in sys.out
    :return: array data with binom distribution.
    '''
    from scipy.stats import binom
    data_binom = binom.rvs(size=size_of_array,n = number_of_trial,p = chance_of_success)
    if print_permission :
        p(data_binom)
    else :
        pass
    if bins != 'f' and kde != 'f' :
        from seaborn import displot
        from matplotlib.pyplot import show
        displot(data_binom,bins=bins,kde=kde)
        show()
    else :
        pass
    return data_binom

def multinomial_distribution(number_of_rows,number_of_trial,chances_of_elements,print_permission=False):
    '''
    :param number_of_rows: number of rows
    :param number_of_trial: sum of provided rows elemets
    :param chances_of_elements: A array that sum of its elements should be 1.length of its demonstrates number of columns
    :param print_permission: print produced data on sys.out
    :return: array data as shape of (number_of_rows,len(chances_of_elements))
    '''
    from numpy.random import multinomial
    data_multinomial = multinomial(size=number_of_rows,n = number_of_trial,pvals=chances_of_elements)
    if print_permission :
        p(data_multinomial)
    else :
        pass
    return data_multinomial

def poisson_distribution(size_of_array,Lambda,print_permission=False,bins='f',kde='f'):
    from numpy.random import poisson
    data_poisson = poisson(lam=Lambda,size=size_of_array)
    if print_permission :
        p(data_poisson)
    else :
        pass
    if bins != 'f' and kde != 'f' :
        from seaborn import displot
        from matplotlib.pyplot import show
        displot(data_poisson,bins= bins,kde=kde)
        show()
    else :
        pass
    return data_poisson

def exponentional_distribution(size_of_the_array,scale,bins='f',kde='f',print_permission=False):
    '''
    The exponential distribution is often used to model the time until an event occurs, such as the time until a failure in a system or the time between arrivals of customers.
    :param size_of_the_array: size of array that is needed
    :param scale:The average or mean of the distribution.
    :param bins:number of devision on range of data
    :param kde:True or False. plot kde or not.
    :param print_permission:print produced data in sys.out
    :return: data array with exponentional distribution.
    '''
    from numpy.random import exponential
    data_exponentional = exponential(size=size_of_the_array,scale=scale)
    if print_permission :
        p(data_exponentional)
    else :
        pass
    if bins != 'f' and kde != 'f':
        from seaborn import displot
        from matplotlib.pyplot import show
        displot(data_exponentional,bins=bins,kde=kde)
        show()
    else :
        pass
    return data_exponentional

def beta_distribution(size_of_array,alpha,Beta,bins='f',kde='f',print_permission=False):
    '''
    :param size_of_array: number of the array's element
    :param alpha: skewness to ward 1
    :param Beta: skewness toward 0
    :param bins: number of devision on range of data
    :param kde: True or False. plot kde or not.
    :param print_permission: True or False. print produced data in sys.out
    :return: data set with beta distribution
    Notice : data mean is alpha/(alpha+Beta)
    '''
    from scipy.stats import beta
    data_beta = beta.rvs(alpha,Beta,size =size_of_array)
    if print_permission :
        p(data_beta)
    else :
        pass
    if kde != 'f' and bins != 'f':
        from seaborn import displot
        from matplotlib.pyplot import show

        displot(data_beta,bins=bins,kde=kde)
        show()
    else :
        pass
    return data_beta

def ks_test(serie1,serie2,print_permission=False):
    '''
    The KS test computes the maximum difference between the empirical cumulative distribution functions (ECDFs) of the two samples.
    :param serie1:The first serie
    :param serie2: The second serie
    :param print_permission: True or False. print produced data in sys.out.
    :return1 statistic: A larger value indicates a greater difference between the two distributions.
    :return2 pvals: This value helps determine the significance of the observed difference. A low p-value (typically less than 0.05) suggests that the two distributions are significantly different.
    '''
    from scipy.stats import kstest
    dic = {}
    keys = ['statistic','pvals']
    score = kstest(serie1,serie2)
    for var in range (0,1+1):
        dic.update({keys[var]:float(score[var])})
    if print_permission :
        p(f'The ks_score for two array distribution similarity is {dic}:')
    else :
        pass
    return dic

def extrem_gradient_boost(data,dependent_column,score_print_permission=False,feature_to_perdict='f'):
    from sklearn import metrics
    from xgboost import XGBRegressor
    data = columns_name_fix(data)
    dependent_column = dependent_column.strip()
    dependent_column = dependent_column.translate(str.maketrans(' .','__'))

    sets = data_preproccessing(data,dependent_column)
    model = XGBRegressor()
    model.fit(sets[0],sets[2])
    training_data_prediction = model.predict(sets[1])
    score = metrics.r2_score(sets[3], training_data_prediction)
    if score_print_permission :
        p(score)
    else :
        pass
    if feature_to_perdict != 'f':
        feature_to_perdict = np.array([feature_to_perdict])
        a = model.predict(feature_to_perdict)
        p(a)
    return 0

def IQR_data(serie,print_permission=False):
    '''
    :return: Q3-Q1 of data
    '''
    IQR = np.quantile(serie,.75)-np.quantile(serie,.25)
    if print_permission :
        p(IQR)
    else :
        pass
    return IQR

def max_data(serie,print_permission=False):
    max_value = serie.max()
    if print_permission :
        p(max_value)
    else :
        pass
    return max_value

def min_data(serie,print_permission=False):
    min_value = serie.min()
    if print_permission :
        p(min_value)
    else :
        pass
    return  min_value

def standard_variation(seire,print_permission=False):
    std_data = np.std(serie)
    if print_permission:
        p(std_data)
    else :
        pass
    return std_data

def skewness_data(serie,print_permission=False):
    Skew = serie.skew()
    if print_permission :
        p(Skew)
    else :
        pass
    return Skew

def kurtosis_data(serie,print_permission=False):
    Kurt = serie.kurt()
    if print_permission :
        p(Kurt)
    else :
        pass
    return Kurt

def plot_kde(serie,bwAdjuster='default',Fill=True,serie_name='data',print_bwAdjuster=False,save_directory=False):
    from seaborn import kdeplot
    from matplotlib.pyplot import figure,title,xlabel,ylabel,show,savefig
    # Calculate bandwidth using Silverman's Rule of Thumb
    if bwAdjuster == 'default' :
        bwAdjuster = 0.9 * min(np.std(serie), (IQR_data(serie) / 1.34)) / (len(serie) ** 0.2)
    else :
        pass
    # Visualize the KDE with the calculated bandwidth
    figure(figsize=(8, 5))
    kdeplot(serie, bw_adjust=bwAdjuster, fill=Fill)
    title(f'The {serie_name} kde distribution')
    xlabel('Value')
    ylabel('Density')
    if save_directory:
        savefig(save_directory, dpi=100, bbox_inches='tight')
    else:
        show()
    return 0

def concatenate_pdfs(input_files, output_file):
    '''input_files: list of directory to the pdfs that must be concatenate'''
    '''output_file: a string that is set a directory for the output file and the directory should contain .pdf'''
    from PyPDF2 import PdfReader, PdfWriter
    # Initialize PdfWriter to store the merged PDF
    writer = PdfWriter()

    # Iterate through each input PDF file
    for file_path in input_files:
        # Read the PDF file
        reader = PdfReader(file_path)
        # Add each page from the input PDF to the writer
        for page in reader.pages:
            writer.add_page(page)

    # Write the merged PDF to the output file
    with open(output_file, "wb") as output:
        writer.write(output)
    
def gregorian_to_jalali(date : str ,return_string=False):
 date_array = date.split('-')
 gy = int(date_array[0])
 gm = int(date_array[1])
 gd = int(date_array[2])
 g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
 if (gm > 2):
  gy2 = gy + 1
 else:
  gy2 = gy
 days = 355666 + (365 * gy) + ((gy2 + 3) // 4) - ((gy2 + 99) // 100) + ((gy2 + 399) // 400) + gd + g_d_m[gm - 1]
 jy = -1595 + (33 * (days // 12053))
 days %= 12053
 jy += 4 * (days // 1461)
 days %= 1461
 if (days > 365):
  jy += (days - 1) // 365
  days = (days - 1) % 365
 if (days < 186):
  jm = 1 + (days // 31)
  jd = 1 + (days % 31)
 else:
  jm = 7 + ((days - 186) // 30)
  jd = 1 + ((days - 186) % 30)
 if return_string:
     return  str(jy)+'-'+str(jm)+'-'+str(jd)
 else :
     return [jy, jm, jd]



def jalali_to_gregorian(date:str ,return_string=False):
 date_array = date.split('-')
 jy = int(date_array[0])
 jm = int(date_array[1])
 jd = int(date_array[2])
 jy += 1595
 days = -355668 + (365 * jy) + ((jy // 33) * 8) + (((jy % 33) + 3) // 4) + jd
 if (jm < 7):
  days += (jm - 1) * 31
 else:
  days += ((jm - 7) * 30) + 186
 gy = 400 * (days // 146097)
 days %= 146097
 if (days > 36524):
  days -= 1
  gy += 100 * (days // 36524)
  days %= 36524
  if (days >= 365):
   days += 1
 gy += 4 * (days // 1461)
 days %= 1461
 if (days > 365):
  gy += ((days - 1) // 365)
  days = (days - 1) % 365
 gd = days + 1
 if ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)):
  kab = 29
 else:
  kab = 28
 sal_a = [0, 31, kab, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
 gm = 0
 while (gm < 13 and gd > sal_a[gm]):
  gd -= sal_a[gm]
  gm += 1
 if return_string:
     return  str(gy)+'-'+str(gm)+'-'+str(gd)
 else :
     return [gy, gm, gd]


def data_characteristic(data,numeric_character=True,class_character=False,
                        data_shape=True,null_percentage=True,columns_null_vals=False,
                        data_columns=False,report_txt_file=False):
    print(f'The data has {data.shape[0]} rows and {data.shape[1]} columns') if data_shape else None
    print(f'The null percentage is {(data.isnull().sum().sum()*100/data.size):0.3f}%') if null_percentage else None 
    print(f'The data has columns named',*(f'*{col}*' for col in data.columns)) if data_columns else None
    print(f'The numeric describtion for your data is:\n',data.describe) if numeric_character else None
    print(f'The class describtion of your data is:\n',data.describe(exclude=[np.number])) if class_character else None
    print(f'Null value counts for each column is:\n',data.isnull().sum()) if columns_null_vals else None

    return 0  


