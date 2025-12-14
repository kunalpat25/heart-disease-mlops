# no heart disease

for i in {1..100}; do
  THALACH=$((100 + RANDOM % 101))
  SLEEP_TIME=$(awk -v min=0.05 -v max=0.8 'BEGIN{srand(); print min+rand()*(max-min)}')

  curl -s -X POST "http://localhost:80/predict" \
    -H "Content-Type: application/json" \
    -d "{\"age\":63,\"sex\":1,\"cp\":3,\"trestbps\":145,\"chol\":233,\"fbs\":1,\"restecg\":0,\"thalach\":$THALACH,\"exang\":0,\"oldpeak\":2.3,\"slope\":0,\"ca\":0,\"thal\":1}"

  sleep "$SLEEP_TIME"
done


## heart disease 

for i in {1..100}; do
  THALACH=$((35 + RANDOM % 66))
  SLEEP_TIME=$(awk -v min=0.05 -v max=0.8 'BEGIN{srand(); print min+rand()*(max-min)}')

  curl -s -X POST "http://localhost:80/predict" \
    -H "Content-Type: application/json" \
    -d "{\"age\":75,\"sex\":1,\"cp\":3,\"trestbps\":145,\"chol\":500,\"fbs\":1,\"restecg\":0,\"thalach\":$THALACH,\"exang\":1,\"oldpeak\":2.3,\"slope\":1,\"ca\":0,\"thal\":1}"

  sleep "$SLEEP_TIME"
done