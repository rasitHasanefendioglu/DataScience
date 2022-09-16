import os.path as path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


class MyData:
    data = pd.DataFrame()
    directory = ""

    def __init__(self, dir_):

        if path.exists(dir_):
            self.directory = dir_
        else:
            raise Exception("Directory is not exist")

    def read_data(self):
        try:
            self.data = pd.read_csv(self.directory)
            return self.data
        except pd.errors.EmptyDataError:
            print("Dataframe is empty control your path or file and try again")
        except:
            print("Some unknown error occurred please make sure you used CSV file")

    def check_reading(self):
        if ~self.data.empty:  # check if the data is read before
            self.read_data()

    def first_n(self, n=20):
        """
        Reads and show the first n rows of the data whose directory given
        :param n:Number wanted to show if user want to select all use -1 for this
        however it doesn't count the last row in this way.
        """
        self.read_data()
        print(self.data.head(n))

    def show_columns(self):
        self.check_reading()
        print(self.data.columns)

    def print_col_row(self):
        self.check_reading()
        col = len(self.data.columns)
        row = len(self.data.index)
        print("Row: ", row, "Column: ", col)
        # print(self.data.shape())

    def get_by(self, group_type, group):
        self.check_reading()
        to_return = self.data[self.data[group_type] == group]
        return to_return

    def sort_by(self, columns):
        self.check_reading()
        sorted_by = self.data.sort_values(by=columns)
        return sorted_by

    def count_by(self, group):
        self.check_reading()
        name, count = np.unique(self.data[group], return_counts=True)
        for i in range(len(name)):
            print(name[i], ":", count[i], end="\t")

    def __del__(self):
        if ~self.data.empty:
            del self.data
        if not self.directory:
            del self.directory
        print("Destructor called MyData object deleted")

    def print_data(self):
        try:
            print(self.data)
        except pd.errors.EmptyDataError:
            print("Data cannot be printed")


class StudentData(MyData):

    def get_stats_by_prep(self):
        self.check_reading()
        prep = self.data[self.data["test preparation course"] == "completed"]
        not_prep = self.data[self.data["test preparation course"] == "none"]
        mean_prep = prep[["math score", "reading score", "writing score"]].mean()
        mean_not_prep = not_prep[["math score", "reading score", "writing score"]].mean()
        min_prep = prep[["math score", "reading score", "writing score"]].min()
        min_not_prep = not_prep[["math score", "reading score", "writing score"]].min()
        max_prep = prep[["math score", "reading score", "writing score"]].max()
        max_not_prep = not_prep[["math score", "reading score", "writing score"]].max()
        print("Mean of preparation students:\n", mean_prep)
        print("Min of preparation students:\n", min_prep)
        print("Max of preparation students:\n", max_prep)
        print("Mean of non-preparation students:\n", mean_not_prep)
        print("Min of non-preparation students:\n", min_not_prep)
        print("Max of non-preparation students:\n", max_not_prep)

    def create_table(self):
        self.check_reading()
        af = [np.min, np.max, np.mean]
        values = ["math score", "reading score"]
        print(self.data.pivot_table(values=values, index="race/ethnicity", columns="gender", aggfunc=af))

    def pair_plot(self):
        self.check_reading()
        pair_data = self.data.loc[:, "test preparation course": "writing score"]
        sns.pairplot(pair_data, height=2.5, hue='test preparation course')
        plt.show()

    def plot_mixed(self, columns, desc, row=2, col=2):
        self.check_reading()
        for i in range(row*col):
            plt.subplot(row, col, i + 1)
            sns.scatterplot(x=columns[0], y=columns[1], data=self.data,
                            hue=desc[i], legend=True)
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
        plt.show()

    def __del__(self):
        if ~self.data.empty:
            del self.data
        if not self.directory:
            del self.directory
        print("Destructor called StudentData object deleted")


class NetflixTitlesData(MyData):

    def __del__(self):
        if ~self.data.empty:
            del self.data
        if not self.directory:
            del self.directory
        print("Destructor called NetflixTitlesData object deleted")

    def show_short_movies(self, print_=True, add_col=False):
        self.check_reading()
        df = self.replace_nan("duration", 21, ["type", "Movie"])
        temp = [int(x.split()[0]) for x in df]
        bool_arr = temp < np.full(len(temp), 20)
        if print_:
            print(bool_arr)
        if add_col:
            to_print = self.get_by("type", "Movie")
            to_print["isShortMovie"] = bool_arr
            print(to_print)
        return bool_arr

    def find_nan(self):
        self.check_reading()
        for col in self.data.columns:
            if self.data[col].isnull().values.any():
                print(col)

    def replace_nan(self, column_, value, condition=None):
        self.check_reading()
        if condition:
            dataf = self.data[self.data[condition[0]] == condition[1]][column_]
        else:
            dataf = self.data[column_]
        dataf.replace("", np.nan, inplace=True)
        dataf.replace(np.nan, str(value), inplace=True)
        return dataf

    def get_by(self, group_type, group):
        self.check_reading()
        temp = self.data[group_type]
        bool_arr = np.full(len(temp), False)
        counter = 0
        for i in temp:
            if group in str(i):
                bool_arr[counter] = True
            counter += 1
        return self.data[bool_arr]

    def plot_hist(self, group_, title_, color=["lightblue", "lightgreen"], kind_="bar"):
        self.check_reading()
        temp = self.data.groupby(group_).size().unstack()
        temp.plot(kind=kind_, title=title_, color=color)
        plt.show()

    def season_count_years(self, condition, hue_="duration", color="plasma"):
        self.check_reading()
        df = self.data[self.data[condition[0]] == condition[1]]
        with sns.axes_style("darkgrid"):
            g = sns.catplot(x="release_year", data=df, aspect=4,
                            kind="count", hue=hue_, order=range(2015, 2020),
                            palette=sns.color_palette(color, 15))
        g.set_xticklabels(step=1)
        plt.show()


class FlightData(MyData):

    def __del__(self):
        if ~self.data.empty:
            del self.data
        if not self.directory:
            del self.directory
        print("Destructor called FlightData object deleted")

    def read_corrupt_data(self):
        try:
            self.data = pd.read_csv(self.directory, skiprows=[0, 1, 3, 4], header=0, nrows=14516, low_memory=False)
            return self.data
        except pd.errors.EmptyDataError:
            print("Dataframe is empty control your path or file and try again")
        except:
            print("Some unknown error occurred please make sure you used CSV file")

    def check_reading(self):
        if ~self.data.empty:  # check if the data is read before
            self.read_corrupt_data()

    def numerical_interpolate(self, cols, method_=None, print_=False):
        self.check_reading()
        df = self.data[cols].interpolate(method=method_)
        if print_:
            print(df)
        return df

    def enumerate_data(self, col, print_=False):
        self.check_reading()
        return_ = list(enumerate(self.data[col].unique()))
        if print_:
            print(return_)
        return return_

    def convert_date(self):
        self.check_reading()
        temp = str(pd.read_csv(self.directory, skiprows=range(1, 14522)).columns).split()[1]
        date_time_str = [temp+x for x in self.data["Time"]]
        date_time_obj = [datetime.datetime.strptime(x, '%m/%d/%Y%I:%M:%S') for x in date_time_str]
        return date_time_obj

    def mean_of_same_cols(self, save=False):
        # TODO fix performance issues
        temp = str(pd.read_csv(self.directory, skiprows=range(1, 14522)).columns).split()
        filename = temp[0]+temp[1].replace("/", "_")
        df = self.data.set_index(list(set(self.data.columns)))
        df = df.groupby(by=df.columns, axis=1).mean()
        df = df.reset_index()
        if save:
            df.to_csv(filename, index=False)
        return df

    def decrease_file_size(self, print_memory_usage=False, print_=True):
        df = self.mean_of_same_cols()
        df = df.dropna(axis=1)
        df = df.dropna()
        if print_memory_usage:
            print("Before adjustments memory usage: ")
            print(self.data.info(memory_usage='deep'))
            print("After adjustments memory usage: ")
            print(df.info(memory_usage='deep'))
        if print_:
            print(df)


# TODO create an interface
flight = 'Documents/Flight.csv'
netflix = 'Documents/netflix_titles.csv'
student = 'Documents/StudentsPerformance.csv'

# For first dataframe
dataStudent = StudentData(student)
dataStudent.read_data()
dataStudent.first_n()
dataStudent.read_data()
dataStudent.show_columns()
sorted_df = dataStudent.sort_by(['writing score', 'reading score'])
print(sorted_df)
dataStudent.count_by("parental level of education")
dataStudent.create_table()
descriptors = ['gender', 'race/ethnicity', 'lunch', 'test preparation course']
column = ["math score", "reading score"]
dataStudent.plot_mixed(columns=column, desc=descriptors)
dataStudent.pair_plot()


# For second dataframe
dataNetflix = NetflixTitlesData(netflix)
test = dataNetflix.show_short_movies(False, True)
print(dataNetflix.get_by("director", "Kirsten Johnson"))
dataNetflix.plot_hist(["release_year", "type"], "Release Count")
dataNetflix.season_count_years(["type", "TV Show"])


# For last dataframe
dataFlight = FlightData(flight)
dataFlight.read_corrupt_data()
dataFlight.print_data()
dataFlight.numerical_interpolate("SELECTED ALTITUDE(MCP)", print_=True)
dataFlight.enumerate_data("Unnamed: 15", True)
print(dataFlight.convert_date())
print(dataFlight.mean_of_same_cols())
dataFlight.decrease_file_size(True, True)
