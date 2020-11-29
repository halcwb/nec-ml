
#r "nuget: Microsoft.ML"
#r "nuget: Microsoft.ML.Vision"
#r "nuget: Microsoft.ML.ImageAnalytics"
#r "nuget: SciSharp.TensorFlow.Redist"

open System
open System.IO

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Vision


[<CLIMutable>]
type ImageData =
    {
        ImagePath : string
        Label : string        
    }


let imgsFolder = Path.Combine(Environment.CurrentDirectory, "NEC")


let files = 
    Directory.GetFiles(imgsFolder, "*.jpeg", SearchOption.AllDirectories)
    |> Array.toList
    |> List.filter (fun s -> s.Contains("Test") |> not)


let isNEC (s : string) = s.Contains("NEC+")
let isNotNec (s : string) = s.Contains("NEC-")


let images =
    files
    |> List.fold (fun acc f ->
        if f |> isNEC || f |> isNotNec then
            {
                ImagePath = f
                Label = if f |> isNEC then "NEC+" else "NEC-"
            }::acc
        else acc

    ) []


let context = MLContext()


let imageDate = 
    context.Data.LoadFromEnumerable(images)
    |> context.Data.ShuffleRows


let testTrainData = context.Data.TrainTestSplit(imageDate, testFraction=0.2)


let validationData = 
    EstimatorChain()
        .Append(context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality=Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue))
        .Append(context.Transforms.LoadRawImageBytes("Image", imgsFolder, "ImagePath"))
        .Fit(testTrainData.TestSet)
        .Transform(testTrainData.TestSet)


let imagesPipeLine =
    EstimatorChain()
        .Append(context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality=Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue))
        .Append(context.Transforms.LoadRawImageBytes("Image", imgsFolder, "ImagePath"))


let imageDataModel = imagesPipeLine.Fit(testTrainData.TrainSet)


let imageDataView = imageDataModel.Transform(testTrainData.TrainSet)


let options =
    let x = ImageClassificationTrainer.Options()
    x.Arch <- ImageClassificationTrainer.Architecture.ResnetV250
    x.Epoch <- 200
    x.BatchSize <- 20
    x.LearningRate <- 0.01f
    x.LabelColumnName <- "LabelKey"
    x.FeatureColumnName <- "Image"
    x.ValidationSet <- validationData
    x


let pipeline =
    EstimatorChain()
        .Append(context.MulticlassClassification.Trainers.ImageClassification(options))
        .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"))



let model = pipeline.Fit(imageDataView)


[<CLIMutable>]
type ImageModelInput =
    {
        Image : byte[]
        LabelAsKey : uint
        ImagePath : string
        Label : string
    }


[<CLIMutable>]
type ImagePrediction =
    {
        ImagePath : string
        Label : string
        PredictedLabel : string
    }


let predictionEngine = context.Model.CreatePredictionEngine<ImageModelInput, ImagePrediction>(model)


let testFiles = 
    Directory.GetFiles(imgsFolder, "*.jpeg", SearchOption.AllDirectories)
    |> Array.toList
    |> List.filter (fun s -> s.Contains("Test") |> not)
    

let testImages =
    testFiles
    |> List.map (fun f ->
        {
            Image = f |> File.ReadAllBytes
            LabelAsKey = if f |> isNEC then 0u else 1u
            ImagePath = f
            Label = if f |> isNEC then "NEC+" else "NEC-"
        }
    )


let testImagesData = context.Data.LoadFromEnumerable(testImages)


let testImageDataView = imagesPipeLine.Fit(testImagesData).Transform(testImagesData)


let predictions = model.Transform(testImageDataView)


let testPredictions = context.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject=false)


testPredictions
|> Seq.iter (fun p ->
    printfn "Label: %s, Predicted: %s" p.Label p.PredictedLabel
)


testImages
|> Seq.iter (fun image ->
    let p = predictionEngine.Predict(image)
    printfn "Path: %s\tLabel: %s, Predicted: %s" p.ImagePath p.Label p.PredictedLabel
)

