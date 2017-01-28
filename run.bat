REM python neural_style.py --content 1-content.jpg --styles 1-style.jpg --output 1-output.jpg --initial-noiseblend 0.0 --style-layer-weight-exp 1.4 --style-weight 1000 --content-weight-blend 0.2 --iterations 1000 --preserve-colors 0  --pooling max

echo %time%
python neural_style.py --content 1-content.jpg --styles 1-style.jpg --output 1-output.jpg --initial-noiseblend 0.0 --style-layer-weight-exp 1.4 --style-weight 1000 --content-weight-blend 0.2 --preserve-colors 0  --pooling max --print-iterations 1 --checkpoint-iterations 1 --iterations 2
echo %time%
