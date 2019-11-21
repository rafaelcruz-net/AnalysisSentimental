using Microsoft.ML;
using SentimentAnalysis.Service;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysis.Service
{
    public static class SentimentAnalysis
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "data.txt");

        private static MLContext Context { get; set; }

        private static ITransformer Model { get; set; }

        private static PredictionEngine<SentimentData, SentimentPrediction> Engine { get; set; }

        static SentimentAnalysis()
        {
            Context = new MLContext();

            TrainTestData splitDataView = LoadData();

            Model = BuildAndTrainModel(splitDataView.TrainSet);
        }

        private static TrainTestData LoadData()
        {
            IDataView dataView = Context.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = Context.Data.TrainTestSplit(dataView, testFraction: 0.55);
            return splitDataView;
        }

        private static ITransformer BuildAndTrainModel(IDataView splitTrainSet)
        {
            var estimator = Context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                                                   .Append(Context.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));

            //Create and Train the model
            var model = estimator.Fit(splitTrainSet);
            return model;

        }

        private static PredictionEngine<SentimentData, SentimentPrediction> GetEngine()
        {
            if (Engine != null)
                return Engine;

            Engine = Context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(Model);

            return Engine;
           
        }

        public static SentimentPrediction GetPrediction(string text)
        {
            var engine = GetEngine();

            return engine.Predict(new SentimentData() { SentimentText = text });
        }
    }
}
