﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SentimentAnalysis.Service
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }
}
