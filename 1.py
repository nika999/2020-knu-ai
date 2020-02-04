{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Prerequisites"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "numpy==1.16.4  \n",
    "pandas==0.25.0  \n",
    "matplotlib==3.1.0  \n",
    "seaborn==0.9.0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### In this notebook you will learn the basics of the main python libraries used for data analysis: \n",
    "\n",
    "    - pandas\n",
    "    - numpy\n",
    "    - matplotlib \n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns \n",
    "import wordcloud\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Working with arrays, Numpy "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'numpy.ndarray'>\n",
      "(3,)\n",
      "1 2 3\n"
     ]
    }
   ],
   "source": [
    "a = np.array([1, 2, 3])   # Create a rank 1 array\n",
    "print(type(a))            # Prints \"<class 'numpy.ndarray'>\"\n",
    "print(a.shape)            # Prints \"(3,)\"\n",
    "print(a[0], a[1], a[2])   # Prints \"1 2 3\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[5 2 3]\n"
     ]
    }
   ],
   "source": [
    "a[0] = 5                  # Change an element of the array\n",
    "print(a)                  # Prints \"[5, 2, 3]\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2, 3)\n",
      "1 2 4\n"
     ]
    }
   ],
   "source": [
    "b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array\n",
    "print(b.shape)                     # Prints \"(2, 3)\"\n",
    "print(b[0, 0], b[0, 1], b[1, 0])   # Prints \"1 2 4\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0. 0.]\n",
      " [0. 0.]]\n",
      "[[1. 1.]]\n",
      "[[7 7]\n",
      " [7 7]]\n",
      "[[1. 0.]\n",
      " [0. 1.]]\n",
      "[[0.16116593 0.1751545 ]\n",
      " [0.80055587 0.59257595]]\n"
     ]
    }
   ],
   "source": [
    "a = np.zeros((2,2))   # Create an array of all zeros\n",
    "print(a)              # Prints \"[[ 0.  0.]\n",
    "                      #          [ 0.  0.]]\"\n",
    "\n",
    "b = np.ones((1,2))    # Create an array of all ones\n",
    "print(b)              # Prints \"[[ 1.  1.]]\"\n",
    "\n",
    "c = np.full((2,2), 7)  # Create a constant array\n",
    "print(c)               # Prints \"[[ 7.  7.]\n",
    "                       #          [ 7.  7.]]\"\n",
    "\n",
    "d = np.eye(2)         # Create a 2x2 identity matrix\n",
    "print(d)              # Prints \"[[ 1.  0.]\n",
    "                      #          [ 0.  1.]]\"\n",
    "\n",
    "e = np.random.random((2,2))  # Create an array filled with random values\n",
    "print(e)                     # Might print \"[[ 0.91940167  0.08143941]\n",
    "                             #               [ 0.68744134  0.87236687]]\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create the following rank 2 array with shape (3, 4)\n",
    "# [[ 1  2  3  4]\n",
    "#  [ 5  6  7  8]\n",
    "#  [ 9 10 11 12]]\n",
    "a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])\n",
    "\n",
    "# Use slicing to pull out the subarray consisting of the first 2 rows\n",
    "# and columns 1 and 2; b is the following array of shape (2, 2):\n",
    "# [[2 3]\n",
    "#  [6 7]]\n",
    "b = a[:2, 1:3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n",
      "77\n"
     ]
    }
   ],
   "source": [
    "# A slice of an array is a view into the same data, so modifying it\n",
    "# will modify the original array.\n",
    "print(a[0, 1])   # Prints \"2\"\n",
    "b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]\n",
    "print(a[0, 1])   # Prints \"77\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Math "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Elementwise operations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = np.array([[1,2],[3,4]], dtype=np.float64)\n",
    "y = np.array([[5,6],[7,8]], dtype=np.float64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 6.  8.]\n",
      " [10. 12.]]\n",
      "[[ 6.  8.]\n",
      " [10. 12.]]\n"
     ]
    }
   ],
   "source": [
    "# Elementwise sum; both produce the array\n",
    "# [[ 6.0  8.0]\n",
    "#  [10.0 12.0]]\n",
    "print(x + y)\n",
    "print(np.add(x, y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-4. -4.]\n",
      " [-4. -4.]]\n",
      "[[-4. -4.]\n",
      " [-4. -4.]]\n"
     ]
    }
   ],
   "source": [
    "# Elementwise difference; both produce the array\n",
    "# [[-4.0 -4.0]\n",
    "#  [-4.0 -4.0]]\n",
    "print(x - y)\n",
    "print(np.subtract(x, y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 5. 12.]\n",
      " [21. 32.]]\n",
      "[[ 5. 12.]\n",
      " [21. 32.]]\n"
     ]
    }
   ],
   "source": [
    "# Elementwise product; both produce the array\n",
    "# [[ 5.0 12.0]\n",
    "#  [21.0 32.0]]\n",
    "print(x * y)\n",
    "print(np.multiply(x, y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.2        0.33333333]\n",
      " [0.42857143 0.5       ]]\n",
      "[[0.2        0.33333333]\n",
      " [0.42857143 0.5       ]]\n"
     ]
    }
   ],
   "source": [
    "# Elementwise division; both produce the array\n",
    "# [[ 0.2         0.33333333]\n",
    "#  [ 0.42857143  0.5       ]]\n",
    "print(x / y)\n",
    "print(np.divide(x, y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.         1.41421356]\n",
      " [1.73205081 2.        ]]\n"
     ]
    }
   ],
   "source": [
    "# Elementwise square root; produces the array\n",
    "# [[ 1.          1.41421356]\n",
    "#  [ 1.73205081  2.        ]]\n",
    "print(np.sqrt(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Vectorized operations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = np.array([[1,2],[3,4]])\n",
    "y = np.array([[5,6],[7,8]])\n",
    "\n",
    "v = np.array([9,10])\n",
    "w = np.array([11, 12])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Inner product of vectors; both produce 219\n",
    "print(v.dot(w))\n",
    "print(np.dot(v, w))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Matrix / vector product; both produce the rank 1 array [29 67]\n",
    "print(x.dot(v))\n",
    "print(np.dot(x, v))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Matrix / matrix product; both produce the rank 2 array\n",
    "# [[19 22]\n",
    "#  [43 50]]\n",
    "print(x.dot(y))\n",
    "print(np.dot(x, y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10\n",
      "[4 6]\n",
      "[3 7]\n"
     ]
    }
   ],
   "source": [
    "x = np.array([[1,2],[3,4]])\n",
    "\n",
    "print(np.sum(x))  # Compute sum of all elements; prints \"10\"\n",
    "print(np.sum(x, axis=0))  # Compute sum of each column; prints \"[4 6]\"\n",
    "print(np.sum(x, axis=1))  # Compute sum of each row; prints \"[3 7]\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Load .zip archive from this link, unzip it and place in the same folder as this notebook:  \n",
    "    https://www.kaggle.com/fizzbuzz/cleaned-toxic-comments"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data \n",
    "df = pd.read_csv(\"train_preprocessed.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>comment_text</th>\n",
       "      <th>id</th>\n",
       "      <th>identity_hate</th>\n",
       "      <th>insult</th>\n",
       "      <th>obscene</th>\n",
       "      <th>set</th>\n",
       "      <th>severe_toxic</th>\n",
       "      <th>threat</th>\n",
       "      <th>toxic</th>\n",
       "      <th>toxicity</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>explanation why the edits made under my userna...</td>\n",
       "      <td>0000997932d777bf</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>train</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>d aww  he matches this background colour i m s...</td>\n",
       "      <td>000103f0d9cfb60f</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>train</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>hey man  i m really not trying to edit war  it...</td>\n",
       "      <td>000113f07ec002fd</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>train</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>more i can t make any real suggestions on im...</td>\n",
       "      <td>0001b41b1c6bb37e</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>train</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>you  sir  are my hero  any chance you remember...</td>\n",
       "      <td>0001d958c54c6e35</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>train</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>congratulations from me as well  use the tool...</td>\n",
       "      <td>00025465d4725e87</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>train</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>cock  suck before you piss around on my work</td>\n",
       "      <td>0002bcb3da6cb337</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>train</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>your vandalism to the matt shirvington article...</td>\n",
       "      <td>00031b1e95af7921</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>train</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>sorry if the word  nonsense  was offensive to ...</td>\n",
       "      <td>00037261f536c51d</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>train</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>alignment on this subject and which are contra...</td>\n",
       "      <td>00040093b2687caa</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>train</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                        comment_text                id  \\\n",
       "0  explanation why the edits made under my userna...  0000997932d777bf   \n",
       "1  d aww  he matches this background colour i m s...  000103f0d9cfb60f   \n",
       "2  hey man  i m really not trying to edit war  it...  000113f07ec002fd   \n",
       "3    more i can t make any real suggestions on im...  0001b41b1c6bb37e   \n",
       "4  you  sir  are my hero  any chance you remember...  0001d958c54c6e35   \n",
       "5   congratulations from me as well  use the tool...  00025465d4725e87   \n",
       "6       cock  suck before you piss around on my work  0002bcb3da6cb337   \n",
       "7  your vandalism to the matt shirvington article...  00031b1e95af7921   \n",
       "8  sorry if the word  nonsense  was offensive to ...  00037261f536c51d   \n",
       "9  alignment on this subject and which are contra...  00040093b2687caa   \n",
       "\n",
       "   identity_hate  insult  obscene    set  severe_toxic  threat  toxic  \\\n",
       "0            0.0     0.0      0.0  train           0.0     0.0    0.0   \n",
       "1            0.0     0.0      0.0  train           0.0     0.0    0.0   \n",
       "2            0.0     0.0      0.0  train           0.0     0.0    0.0   \n",
       "3            0.0     0.0      0.0  train           0.0     0.0    0.0   \n",
       "4            0.0     0.0      0.0  train           0.0     0.0    0.0   \n",
       "5            0.0     0.0      0.0  train           0.0     0.0    0.0   \n",
       "6            0.0     1.0      1.0  train           1.0     0.0    1.0   \n",
       "7            0.0     0.0      0.0  train           0.0     0.0    0.0   \n",
       "8            0.0     0.0      0.0  train           0.0     0.0    0.0   \n",
       "9            0.0     0.0      0.0  train           0.0     0.0    0.0   \n",
       "\n",
       "   toxicity  \n",
       "0       0.0  \n",
       "1       0.0  \n",
       "2       0.0  \n",
       "3       0.0  \n",
       "4       0.0  \n",
       "5       0.0  \n",
       "6       4.0  \n",
       "7       0.0  \n",
       "8       0.0  \n",
       "9       0.0  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Explore a few lines from the table\n",
    "df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>comment_text</th>\n",
       "      <th>id</th>\n",
       "      <th>identity_hate</th>\n",
       "      <th>insult</th>\n",
       "      <th>obscene</th>\n",
       "      <th>severe_toxic</th>\n",
       "      <th>threat</th>\n",
       "      <th>toxic</th>\n",
       "      <th>toxicity</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>159566</th>\n",
       "      <td>and for the second time of asking  when your ...</td>\n",
       "      <td>ffe987279560d7ff</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>159567</th>\n",
       "      <td>you should be ashamed of yourself that is a ho...</td>\n",
       "      <td>ffea4adeee384e90</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>159568</th>\n",
       "      <td>spitzer umm  theres no actual article for pros...</td>\n",
       "      <td>ffee36eab5c267c9</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>159569</th>\n",
       "      <td>and it looks like it was actually you who put ...</td>\n",
       "      <td>fff125370e4aaaf3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>159570</th>\n",
       "      <td>and i really don t think you understand i ca...</td>\n",
       "      <td>fff46fc426af1f9a</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                             comment_text                id  \\\n",
       "159566   and for the second time of asking  when your ...  ffe987279560d7ff   \n",
       "159567  you should be ashamed of yourself that is a ho...  ffea4adeee384e90   \n",
       "159568  spitzer umm  theres no actual article for pros...  ffee36eab5c267c9   \n",
       "159569  and it looks like it was actually you who put ...  fff125370e4aaaf3   \n",
       "159570    and i really don t think you understand i ca...  fff46fc426af1f9a   \n",
       "\n",
       "        identity_hate  insult  obscene  severe_toxic  threat  toxic  toxicity  \n",
       "159566            0.0     0.0      0.0           0.0     0.0    0.0       0.0  \n",
       "159567            0.0     0.0      0.0           0.0     0.0    0.0       0.0  \n",
       "159568            0.0     0.0      0.0           0.0     0.0    0.0       0.0  \n",
       "159569            0.0     0.0      0.0           0.0     0.0    0.0       0.0  \n",
       "159570            0.0     0.0      0.0           0.0     0.0    0.0       0.0  "
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Same as previous, but from the end of the file, defaul number of lines = 5\n",
    "df.tail() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Documentation   \n",
    "\n",
    "Refer to the documentation from this link to find some information about working with pandas DataFrames:  \n",
    "    https://pandas.pydata.org/pandas-docs/version/0.25.0/reference/api/pandas.DataFrame.html "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Please, show which columns are available in this dataframe (For example: 'comment_text', 'id', ...):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['comment_text' 'id' 'identity_hate' 'insult' 'obscene' 'set'\n",
      " 'severe_toxic' 'threat' 'toxic' 'toxicity']\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "df = pd.read_csv(test_preprocessed.cvs)\n",
    "print(df.columns.values)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Please, show the DataFrame's shape (rows, columns) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
   {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(159571, 10)\n"
     ]
    }
   ],
   "source": [
    "print(df.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Calculate how much commens are labelled as:\n",
    "\n",
    " 1. Identity hate message \n",
    " 2. Insult message\n",
    " 3. Obscene message  \n",
    "etc... \n",
    " 6. Toxic message. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
   {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "identity_hate     1405.0\n",
      "insult            7877.0\n",
      "obscene           8449.0\n",
      "severe_toxic      1595.0\n",
      "threat             478.0\n",
      "toxic            15294.0\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "labels = df[['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic']].sum()\n",
    "print(labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Pre-process dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You can make our DataFrame smaller to make it easier to work with it: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_sample = df.sample(n=1000) # random selection \n",
    "df_small = df[:100] # select the first 100 rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "comment_text      object\n",
       "id                object\n",
       "identity_hate    float64\n",
       "insult           float64\n",
       "obscene          float64\n",
       "set               object\n",
       "severe_toxic     float64\n",
       "threat           float64\n",
       "toxic            float64\n",
       "toxicity         float64\n",
       "dtype: object"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the data type\n",
    "df_sample.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "number of duplicate rows:  (0, 9)\n"
     ]
    }
   ],
   "source": [
    "# Check duplicated rows and delete them if any\n",
    "duplicate_rows_df = df[df.duplicated()]\n",
    "\n",
    "print(\"number of duplicate rows: \", duplicate_rows_df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop unnecessary columns:\n",
    "\n",
    "df.drop(columns='set', inplace=True, errors='ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualizations"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Histogram plot "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Count label occurences\n",
    "\n",
    "labels = df[['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic']].sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "identity_hate     1405.0\n",
       "insult            7877.0\n",
       "obscene           8449.0\n",
       "severe_toxic      1595.0\n",
       "threat             478.0\n",
       "toxic            15294.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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4ospKSlh//3354033qjM27hx48oqy169elW7ry+++IIzzzyTkpISDj74YGbPnl2QYzLbmg0ePJj999+f/v37s2TJEgAOOOAARowYwapVq5g1axaTJk1izpw5leucc845dO3alRtvvJGIWGebc+fOZY899qicLy4uZu7cuYU/GLMCqu8agkqS2gPfBl4DWkfEvLRoPtA6TRcBc3JWK09pNaVXt5/zJU2UNHHhwoW1Vv4KNVVbzpkzhxdffJF27dpVpr3wwgvMmDGDGTNmMGTIEC644ILKZTvuuGNlleWIESOq3df999/PrrvuysyZM7nsssu45pprav14zLZmF1xwAe+++y5lZWW0adOGK664AsgC9+LiYkpLS7n00ks59NBDady4MZA9Lpg8eTKvvPIKr7zyCo888kh9HoJZndkiAgJJOwNPA5dGxMe5yyILz9cN0TdRRAyJiNKIKG3ZsmVtbbZSTdWWl112GbfeeiuSKtOGDx/OWWedhSS6d+/O0qVLmTdv3jrr1mT48OH065c1lzjttNMYO3ZstXczZtuq1q1b07hxYxo1asR5551XWa3fpEkTBg0aRFlZGcOHD2fp0qXss88+ABQVZfcSX//61/m3f/u3ah8FFBUVrVWjUF5eXrme2daq3gMCSV8jCwYei4g/pOQP06MA0ndFM9+5wB45qxentJrStwjDhw+nqKiIAw44YK309VU7fv7555SWltK9e3eeffbZarebu36TJk1o1qwZH330UYGOwmzrkxtgP/PMM5WP8lasWMGnn34KwJgxY2jSpAmdOnVi1apVlY/0vvzyS55//vlqH//16tWLYcOG8cUXXzBr1ixmzJjBQQcdVAdHZFY4Tepz58pul+8HpkXEf+csGgH0A25O38Nz0i+SNIysAeGyiJgnaTTwq5yGhEcDP6uLY9iQFStW8Ktf/YoXX3xxo9Z7//33KSoq4r333qNnz5506dKFvffeu0ClNNv69e3bl3HjxrFo0SKKi4v5z//8T8aNG0dZWRmSaN++Pffeey8ACxYs4JhjjqFRo0YUFRVVPhb44osvOOaYY/jyyy9ZvXo1Rx11FOeddx4AI0aMYOLEidxwww107tyZM844g06dOtGkSRPuuuuuykcOZlureg0IgMOAHwKTJZWltP8gCwSelDQAeB84Iy0bCRwPzARWAOcARMRiSTcCr6d8N0TE4ro5hPV79913mTVrVmXtQHl5Od26dWPChAnrrXas+N5rr73o0aMHb7755joBQcX6xcXFrFq1imXLlrHbbrvV0ZGZbVkef/zxddIGDBhQbd727dszffr0ddJ32mknJk2aVO06vXr1WquB789//nN+/vOfb2JpzbY89RoQRMRfAdWw+Mhq8gdwYQ3begB4oPZKVzu6dOmyVscm7du3Z+LEiey+++706tWLwYMH06dPH1577TWaNWtGmzZtWLJkCU2bNmX77bdn0aJF/O1vf+Pqq69eZ9u9evVi6NChHHLIITz11FP07NlzrTYKZlujSeedX99FqFMH3jekvotgBtR/DUGDU121ZU13KccffzwjR46kpKSEpk2b8uCDDwIwbdo0fvSjH9GoUSPWrFnDwIED6dSpEwC/+MUvKC0tpVevXgwYMIAf/vCHlJSU0KJFC4YNG1Znx2lmZg2LA4JaVl21Za7cvgIkcdddd62T59BDD2Xy5MnVrn/DDTdUTu+www78/ve/37SCmpmZ5XBAUIMf3PHH+i5CnXr0khPquwhmZlaP6v21QzMzM6t/DgjMzMzMAYGZ2bZu9erVfPvb3+bEE08E4PDDD68cS6Vt27acfHI2nMyyZcs46aSTOOCAA+jcuXNlQ+iqJk2aRJcuXSgpKeHiiy92D6pbCQcEZmbbuDvuuINvfetblfOvvPJK5VgqhxxyCKeeeioAd911F506deKtt95i3LhxXHHFFaxcuXKd7V1wwQXcd999lWO1VDe+i215HBCYmW3DysvL+eMf/8i55567zrKPP/6Yl19+ubKGQBLLly8nIvjkk09o0aIFTZqs3TZ93rx5fPzxx3Tv3h1JnHXWWTV2v25bFr9lYGa2Dbv00ku59dZbWb58+TrLnn32WY488kh22WUXAC666CJ69epF27ZtWb58OU888QSNGq19Xzl37lyKi4sr5z009NbDNQRmZtuo559/nlatWnHggQdWu/zxxx+nb9++lfOjR4+ma9eufPDBB5SVlXHRRRfx8ccfV7uubX0cEJiZbaP+9re/MWLECNq3b0+fPn14+eWX+cEPfgDAokWLmDBhAiec8FUfJQ8++CCnnnoqkigpKaFDhw784x//WGubRUVFlJeXV857aOithwMCM7Nt1K9//WvKy8uZPXs2w4YNo2fPnjz66KMAPPXUU5x44ons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\n",
      "text/plain": [
       "<Figure size 576x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (8, 4))\n",
    "ax = sns.barplot(labels.index, labels.values, alpha = 0.8)\n",
    "plt.title(\"# per class\")\n",
    "plt.ylabel('# of occurrences', fontsize = 12)\n",
    "plt.xlabel('Type ', fontsize = 12)\n",
    "\n",
    "# Add text labels\n",
    "rects = ax.patches\n",
    "labels = labels.values\n",
    "for rect, label in zip(rects, labels):\n",
    "    height = rect.get_height()\n",
    "    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha = 'center', va = 'bottom')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Comment what you see here: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "Большинство комментариев относятся к категории 'toxic', половина из них к категориям 'obscene' и 'insult', и наименьшее количество комментариев из категорий 'severe_toxic', 'identity_hate' и  'threate'."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### See how labels correlate with each other: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fedc13a2be0>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "temp_df = df[['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic']]\n",
    "\n",
    "corr = temp_df.corr()\n",
    "plt.figure(figsize = (10, 8))\n",
    "sns.heatmap(corr,\n",
    "            xticklabels = corr.columns.values,\n",
    "            yticklabels = corr.columns.values, annot=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The above plot indicates a pattern of co-occurance. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Comment what you see from the above picture:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "По этой картинке мы можем проследить попарную зависимость между некоторыми категориями."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Wordclouds"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Calculate the number of the uniq words in all of the comments. (Tip: to split text on words use text.split() command, it will separate your text by space)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
   {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "151924\n"
     ]
    }
   ],
   "source": [
    "def calculate_uniq(df,word):\n"
    "    wtext = \"\"\n",
    "    for text in df['comment_text']:\n",
    "        wtext += text\n",
    "        word = wtext.split()\n",
    "    return word\n",
    "print(len(set(a)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's work with wordclouds. \n",
    "The next task would be to select all of the words from the textual data and create a wordcloud. \n",
    "Here you can see an example of such visualisation: \n",
    "\n",
    "https://towardsdatascience.com/word-clouds-in-python-comprehensive-example-8aee4343c0bf \n",
    "\n",
    "Create the same visualization for our dataset. \n",
    "Describe what you see."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
   {
     "data": {
      "text/plain": [
       "<Figure size 1440x1080 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from os import path\n",
    "from PIL import Image\n",
    "from wordcloud import WordCloud, STOPWORDS\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "text = \"\".join(df.comment_text)\n",
    "wordcloud = WordCloud(width=2000, height=1000, contour_color=\"black\", max_words=500, \n",
    "                      relative_scaling = 0, background_color = \"white\").generate(text)\n",
    "plt.figure(figsize=[20,15])\n",
    "plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation=\"bilinear\")\n",
    "plt.axis(\"off\")\n",
    "_=plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Distributions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The main goal of this task is to plot words distributions for each category. \n",
    "What does it mean: \n",
    "\n",
    "1. You need to select words from each category (identity_hate, insult, etc.) \n",
    "2. Plot a historgram with the most popular words for each category \n",
    "3. Try to delete stop words: \n",
    "    1. Install nltk library\n",
    "    2. from nltk.corpus import stopwords - in the stopwords you will see the most common stopwords \n",
    "    3. Filter them from the words for each category\n",
    "4. Plot a histogram again. Has it changed? \n",
    "5. Analyse received results. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
   {
     "data": {
      "text/plain": [
       "<Figure size 1080x1440 with 6 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Рисуем гистограммы, выбирая самые популярные слова с каждой категории\n",
    "import re\n",
    "import nltk\n",
    "import seaborn as sns\n",
    "from collections import Counter\n",
    "from nltk.tokenize import sent_tokenize, word_tokenize\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "category = ['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic', 'toxicity']\n",
    "for i in category:\n",
        "text = ""\n",
        "for j in range(len(df[i])):\n",
            "if df[i][j] != 0:\n",
                "text = text + " " + (df.comment_text[j])\n",
    "cnt = Counter(re.split(r'\s+', text)).most_common(13)\n",
    "data = {'words', 'frequency'}\n",
    "df_i = pd.DataFrame(cnt,columns = data)\n",
    "plt.figure(figsize = (8, 4))\n",
    "sns.barplot(y = 'words', x = 'frequency', data = df_i)\n",
    "plt.title('Frequency', fontsize = 10)\n",
    "plt.ylabel('Occurrences', fontsize = 10)\n",
    "plt.xlabel(i, fontsize = 10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
   {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x1440 with 6 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Гистограммы после фильтрации слов\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "stopwords.words('english')\n",
    "category = ['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic', 'toxicity']\n",
    "for i in category:\n",
    "    text = ""\n",
    "    for j in range(len(df[i])):\n",
    "        if df[i][j] != 0:\n",
    "            text = text + " " + (df.comment_text[j])\n",
    "stop_words = set(stopwords.words('english'))\n",
    "wordtokens = word_tokenize(text)\n",
    "filtr = [] \n",
    "for w in wordtokens:\n",
        "if w not in stop_words:\n",
            "filtr.append(w)\n",
    "cnt = Counter(filtr).most_common(13)\n",
    "data = {'words', 'frequency'}\n",
    "df_i = pd.DataFrame(cnt, columns = data)\n",
    "plt.figure(figsize = (8, 4))\n",
    "sns.barplot(y = 'words', x = 'frequency', data = df_i)\n",
    "plt.title('Frequency', fontsize = 10)\n",
    "plt.ylabel('Occurrences', fontsize = 10)\n",
    "plt.xlabel(i, fontsize = 10)\n",
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Please, write down what did you learn and find during this task.  \n",
    "What was the most difficult part?  \n",
    "What did you enjoy?  \n",
    "Suggest your improvements.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "Я выучила новые библиотеки, научилась работать с таблицами, строить гистограммы и анализировать данные.\n",
    "Самой сложной частью была 'Distributions'.\n",
    "Я получила много полезных навыков, что считаю большим плюсом."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
