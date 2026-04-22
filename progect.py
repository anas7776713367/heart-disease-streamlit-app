import pandas as pd  
import numpy as np  
import streamlit as st 
import seaborn as sns
import matplotlib.pyplot as plt
rtl="""
<style>
   main block-container{
   direction: RTL;
   text-align:right;}
   [data-testid="stSidebar"]{
       direction:RTL;
       text-align:right;
       }
    h1,h2,h3p,div,label{
       text-align:right;
       font-family:'Tahoma',sans-serf;
       
       }
</style>"""



st.markdown(rtl,unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #E63946;'>❤️ نظام التشخيص الذكي والتنبؤ بأمراض القلب</h1>", unsafe_allow_html=True)
st.markdown("---")
st.image("titl_header.jpg")
st.markdown("---")
st.markdown("<h2 style='color: #2E86C1; font-family:tahoma'>فكرة وأهداف المشروع</h2>", unsafe_allow_html=True)
st.markdown("""
            <p>بناءً على التوجهات الحديثة في دمج التكنولوجيا بالقطاع الصحي، يأتي هذا المشروع كخطوة نحو تعزيز الرعاية الصحية الرقمية. </p>
<p><em style="color:red;text-align:center;font-size:40px">:الهدف العام</em></p>
<p>تطوير نظام خبير قادر على تقديم تقييم أولي لمخاطر الإصابة بأمراض القلب باستخدام خوارزميات الذكاء الاصطناعي، مما يساهم في تقليل عبء الفحوصات اليدوية وتوفير تشخيص مبكر دقيق.</p>
<h2 style="color:red ">:المميزات الرئيسية</h2>      
<p> <strong style="color:green">دقةعالية :</strong> يعتمد النظام على نماذج رياضية مدربة على بيانات طبية حقيقية.</p>
<p><strong style="color:green"> سرعة الاستجابة:</strong> تحليل فوري للمؤشرات الحيوية وتقديم النتيجة في ثوانٍ</p>
<p><strong style="color:green"> واجهة مستخدم سهلة:</strong>  : تصميم يراعي سهولة الاستخدام للمختصين والمرضى على حد سواء.</p>
     

""",unsafe_allow_html=True)

st.markdown("---")
with st.expander(" التفاصيل التقنية والمنهجية"):
    st.markdown("""
    ### 1. مصدر البيانات (Dataset):
    تم استخدام قاعدة بيانات **Cleveland Heart Disease**، وهي واحدة من أدق قواعد البيانات المتاحة للجمهور، وتحتوي على متغيرات سريرية مثل العمر، ضغط الدم، ومعدل ضربات القلب.

    ### 2. معالجة البيانات (Preprocessing):
    تم تنظيف البيانات ومعالجتها باستخدام مكتبات **Pandas** و **NumPy** لضمان خلوها من القيم المفقودة وتحويل المتغيرات النصية إلى قيم رقمية يفهمها النموذج.
    

    ### 3. تقنيات الذكاء الاصطناعي:
    تمت مقارنة عدة خوارزميات تصنيف، واختيار الخوارزمية الأكثر استقراراً ودقة (  **Logistic Regression** )  تم استخدام خوارزمية  اللوجستي .

    ### 4.بيئة التطوير:
    * **لغة البرمجة:** Python .
    * **واجهة المستخدم:** Streamlit .
 
    """,unsafe_allow_html=True)
st.subheader("قم بادخال بيانات المريض ليقوم النموذج بالتنبؤ")
    

col1,col2,col3=st.columns([4,4,4])

#تقسيم البيانات 
data=pd.read_csv(r"Data.csv")
data.insert(0,"عمود الواحدات",1)
#@st.cache_data
def split_data():
    st.markdown("---")
    if st.sidebar.checkbox("عرض باينات النموذج"):
        rwos=st.number_input("عدد الصفوف",5,data.shape[0])
        st.dataframe(data.head(rwos))
    col=data.shape[1]
    
    x_train=data.iloc[:,0:col-1].values
    st.markdown("---")
    if st.sidebar.checkbox("عرض بيانات التعلم"):
        st.subheader("مصفوفة بيانات التعلم")
        st.dataframe(x_train)
   
    y_trin=data.iloc[:,col-1:].values
    if st.sidebar.checkbox("عرض بيانات التشخيص"):
        st.markdown("---")
        st.subheader("مصفوفة بيانات الهدف")
        st.dataframe(y_trin)

    wheghet=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    wheghet=wheghet.reshape(-1,1)
    if st.sidebar.checkbox("عرض الاوزان"):
        st.markdown("---")
        st.subheader("مصفوفة الاوزان")
        st.dataframe(wheghet.reshape(-1))
    return x_train,y_trin,wheghet
X_trin,Y_train,w=split_data()

def function_train(x,w):
        z=np.dot(x,w)
        predict=1/(1+np.exp(-z))
        predict=(predict>=0.5).astype("int")        
        return predict

#دالة التكلفة 
@st.cache_data
def cost(x,y,w,alpha,n):
    cst=[]
    with st.spinner("جاري تدريب النموذج"):
        for i in range(n):
            z=np.dot(x,w)
            predict=1/(1+np.exp(-z))
            cst_list=np.sum((y.T @np.log(predict)+(1-y).T @ np.log(1-predict)))*-1/len(y)
            cst.append(cst_list)
            
            erorr=predict-y
            dw=1/len(x)*np.dot(x.T,erorr)
            w=w-alpha*dw
        return w ,cst

alpha=0.0001
n=100000
if st.sidebar.button("تدريب النموذج") :    
      new_w=cost(X_trin,Y_train,w,alpha,n)     
new_w,cst=cost(X_trin,Y_train,w,alpha,n)

#دالة تعديل الاوزان باستخدام الدالة العادية
def normal():
    revrs=np.linalg.pinv(np.dot(X_trin.T,X_trin))
    m=np.dot(X_trin.T,Y_train)
    normaltion=np.dot(revrs,m)
    return normaltion
normal_w=normal()

if st.sidebar.checkbox("عرض الاوزان بعد التدريب"):
  with  col3:
    st.markdown("---")
    st.subheader("الاوزان بعد التعلم")
    st.dataframe(new_w)
colom1,colum2,colum3=st.columns([2,4,2])
st.markdown("---")


def enter():
    
        with col1:
            
            data1=st.number_input("العمر",0.0,1000.0)
            data2 = st.selectbox("الجنس", ["ذكر", "انثى"])
            data2 = 1 if data2=="ذكر" else 0
            
   
        
            data3=st.number_input("الام الصدر",0.0,1000.0)
            data4=st.number_input("ضغط الدم",0.0,1000.0)
        with col2:
            data5=st.number_input("الكولسترول",0.0,1000.0)
            data6=st.number_input("السكر",0.0,1000.0)
            data7=st.number_input("تخطيط القلب",0.0,1000.0)
            data8=st.number_input("اقصى النبض",0.0,1000.0)
            
        with col3:
            data9=st.number_input("الذبحة الرياضية",0.0,1000.0)
            data10=st.number_input("STانخفاض",0.0,1000.0)
            data11=st.number_input("ميل القطعة",0.0,1000.0)
            data12=st.number_input("الاوعية الرئيسية",0.0,1000.0)
        with col2:
            data13=st.number_input("الثلاسيميا",0.0,1000.0)
        new_data=np.array([1,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13])
        return new_data


#حقل ادخال البيانات

with colum2:
    try:
        
            
            a=enter()
            if st.button("ارسال البانات",use_container_width=True,type="primary"):

                st.subheader(" التوقع من البيانات الجديدة")
                perd=function_train(a,new_w)
            
            if (perd==1).all():
                st.balloons()
                st.success("الناتج سليم")
                st.image("توقع_سليم.jpg")
            else:
                st.success("الناتج مصاب")
                st.image("توقع_سيء.jpg")    
        
    except Exception as e:
                #st.error("خطا في الادخال:يرجى التاكد من ان جميع القيم ضمن النطاق الصحيح")
            st.info("يرجي املاء جيع الحقول التالية بالبيانات الناسبة لكل حقل ")

#دالة لحساب دقة النموذج
def acuorcy():
    acr=function_train(X_trin,new_w)
    acor=np.sum(acr==Y_train)/len(Y_train)*100
    st.subheader("دقة النموذج")
    m={"الدقة":acor}
    mn=pd.Series(m)
    st.write(mn)
    
if st.sidebar.checkbox("دقة النموذج بعد التعلم"):
    acuorcy()
    st.markdown("---")

 #الرسومات التوضيحية  
if st.sidebar.checkbox("تكلفة النموذج"):   
    st.subheader("تكلفة النموذج")
    mn=pd.DataFrame([cst[n-1]],[0],columns=["التكلفة النهائية"])
    mn["التكلفة الاولية"]=cst[0]
    st.dataframe(mn)
    fig,ax=plt.subplots()
    plt.plot(cst,label="The Cost")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)
c1=st.columns(4)
c2=st.columns(4)
c3=st.columns(4)
c4=st.columns(1)

st.markdown("""
<div style='text-align:center; margin-top:30px;'>
    <hr>
    <p> تم تطوير هذا المشروع بواسطة: <b>انس محمد عبد الجليل</b></p>
    <p> قسم الذكاء الاصطناعي - جامعة تعز</p>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center;'>
🔗 رابط المشروع: https://your-app.streamlit.app
</p>
""", unsafe_allow_html=True)

