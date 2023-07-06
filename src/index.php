<?php

declare(strict_types=1);

// namespace MachineLearning;
require __DIR__ . '/../vendor/autoload.php';

use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Transformers\NumericStringConverter;

$caminho = __DIR__.'/arc-teste.csv';

$csv = new CSV($caminho, true);
$dataset = Labeled::fromIterator($csv);

$dataset
    ->apply(new NumericStringConverter())
    ->apply(new LambdaFunction(
        function ( array &$samples){
            unset($samples[0]);
            $samples =  array_values($samples);
        }
    ));

[$training, $testing] = $dataset->stratifiedSplit(0.8);    

$tree = new RandomForest(new ClassificationTree());
$tree->train($training); 

$predictions = $tree->predict($testing);

$confusionMatrix = new ConfusionMatrix();

$retorno = $confusionMatrix->generate($predictions, $testing->labels());

echo '<PRE>';
print_r($retorno);
echo '</PRE>';

$accuracy =  new Accuracy();

$_accuracy = $accuracy->score($predictions, $testing->labels());

echo '<PRE>';
print_r($_accuracy);
echo '</PRE>';

$_tree = $tree->predict(new Unlabeled([
    [10_000, 2009],
    [100_000, 1995]
]));

echo '<PRE>';
print_r($_tree);
echo '</PRE>';