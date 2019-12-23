# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:56:47 2019

@author: Jing Yang (jyang7@fsu.edu) and Ming Ye (mye@fsu.edu). This program was modified from the python codes
available at https://salib.readthedocs.io/en/latest/api.html. 

The MIT License (MIT)

Copyright (c) 2019 Jing Yang and Ming Ye

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import pandas as pd

class ResultDict(dict):
    '''Dictionary holding analysis results.

    Conversion methods (e.g. to Pandas DataFrames) to be attached as necessary
    by each implementing method
    '''
    def __init__(self, *args, **kwargs):
        super(ResultDict, self).__init__(*args, **kwargs)

    def to_df(self):
        '''Convert dict structure into Pandas DataFrame.'''
        return pd.DataFrame({k: v for k, v in self.items() if k is not 'names'},
                            index=self['names'])