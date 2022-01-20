from django.shortcuts import render
from django.http import JsonResponse
from .models import PredResults
from sklearn.preprocessing import StandardScaler
import pickle

# Create your views here.

MODEL_PATH = 'artifacts/model.pkl'
NORMALIZE = pickle.load(open('artifacts/std.pkl','rb'))

def predict(request):
    return render(request, 'predict.html')


def predict_chances(request):
    if request.POST.get('action') == 'post':
        #recieve data from client
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))
        model = pickle.load(open(MODEL_PATH,'rb'))
        x = NORMALIZE.transform([[sepal_length,sepal_width,
                                 petal_length,petal_width
                                 ]])
        result = model.predict(x)
        classification = ['setosa', 'versicolor', 'virginica'][result[0]]
        PredResults.objects.create(sepal_length=sepal_length,
                                   sepal_width=sepal_width,
                                   petal_length = petal_length,
                                   petal_width=petal_width,
                                   classification=classification
        )
    return JsonResponse({'result':classification,
                         'sepal_length':sepal_length,
                         'sepal_width':sepal_width,
                         'petal_length':petal_length,
                         'petal_width':petal_width
                         }, safe=False)


def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)