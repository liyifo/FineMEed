# FineMed

Medication Mapping and Diagnosis Enhancement for Fine-grained Medication Recommendation

## Environment

```python
python==3.9.18
torch==2.1.1
tqdm==4.66.1
dgl==1.1.2.cu118
scikit-learn==1.3.2
```
You can build the conda environment for our experiment using the following command:
```
conda env create -f environment.yml
```

## Prepare the datasets

We use the private dataset SMH and [MIMIC-III](https://mimic.mit.edu/docs/iii/) v1.4 for our experiments. We have tried our best to correct many of the issues mentioned in the paper in the MIMIC-III data. Once the paper is accepted, we will publicly disclose some of the data from the SMH dataset, or later to provide a platform that can train and validate online.

If the above conditions are ready, run the following commands in sequence to preprocess:

```shell
# preprocess the dataset(mimic-iii)
cd data

# Generate a new smiles file
python get_smiles.py

# preprocess data
python preprocess_mimic-iii.py

# Restore the codes in the dataset to standard ICD-9 codes
python icd_process.py

# Separate different chapters of clincal notes
python segment.py

```

## Diagnosis Enhancement

Please use the following code for Diagnosis Enhancement. The results of LLM is somewhat random and may not be output in the given format, requiring repeated runs, which we did about 3-4 times.

```shell
cd ./diagnosis_enhancement

# Need to set api in this file
python corresponder.py

# Need to set api in this file
python checker.py
```

## Medication Mapping

```shell
cd ./medication_mapping

# Need to set api in this file
python corresponder.py

# Need to set api in this file
python checker.py
```

The choice of the LLM can greatly affect the accuracy of the reasoning, please use the GPT4o model. We spent at least $1400 on Diagnosis Enhancement and Medication Mapping. We are using it for further work about reasonability, and when it is done we will make this part of the data freely available.

## Train or Test


You can train or test the model using the following command:
```python
# process data
cd ./data
python input_pre.py
python input_generate.py
cd ..

# mimic-iii
python main_mimic-iii.py

# test, please set resume_path to the best model.
python main_mimic-iii.py --test

```

## About SMH dataset

We have used only a small portion of the available data in this paper. We are preparing this dataset and will make it fully public in the near future. The SMH dataset contains at least 10,000 pieces of high-quality data on the diagnosis and treatment process from both traditional Chinese medicine and modern medicine perspectives.

An unprocessed, simple example of raw data, just for reference:

> 姓名：XXX
> 入院时间：20XX年02月XX日 10时39分
> 性别：X
> 出院时间：20XX年03月XX日 15时17分
> 年龄：XX岁
> 住院天数：11天
> 入院情况：
>     患者因“阵发性胸闷、憋气3月，加重5天。”入院。患者因“阵发性胸闷、憋气3月，加重5天。”入院。患者3月前无明显诱因出现阵发性胸闷、憋气，情绪变化及活动后自觉胸闷、憋气症状明显加重，体力活动明显受限，休息约5-10分钟可逐渐自行缓解，曾多次于我科因“急性心力衰竭”住院治疗，与强心、利尿、抗感染等对症治疗后，症状好转出院。出院后规律服用相关药物治疗，疗效尚可。5天前患者患者未规律服药自觉胸闷、憋气症状再次发作，双下肢乏力，体力活动明显受限，稍有活动即觉胸闷、憋气症状明显加重，休息约10-15分钟可逐渐自行缓解，未予特殊诊疗。现患者为求进一步中西医结合诊疗，入住我病区。入院症见：胸闷憋气、心悸，伴气短、汗出，乏力，偶腰酸腰痛，踝关节痛，后背皮肤瘙痒，口干，偶有恶心、腹胀，无发热，无咳嗽咳痰，无腹痛，无头晕耳鸣，无眼干，纳一般，眠欠佳，二便调。查体：体温：36.2℃  脉搏：82次/分  呼吸：18次/分  血压：100/67mmHg（L）102/64mmHg（R）  Padua评分：1分（低危），患者青年，男,发育正常，营养良好，神志清楚，步入病房,查体合作，全身皮肤及粘膜无黄染,未见皮下出血,浅表淋巴结未及肿大。头颅五官无畸形，眼睑无水肿,巩膜无黄染,双侧瞳孔等大等圆，对光反射灵敏，外耳道无异常分泌物，鼻外观无畸形，口唇红润，伸舌居中，双侧扁桃体正常，表面未见脓性分泌物，颈软，无抵抗感，双侧颈静脉正常，气管居中，甲状腺未及肿大，未闻及血管杂音。胸廓正常,双肺呼吸音清晰，未闻及干、湿啰音，未闻及胸膜摩擦音。心界扩大，心率82次/分，心律齐整，心音可,各瓣膜听诊区未闻及杂音，未闻及心包摩擦音。脉搏规整，无水冲脉、枪击音、毛细血管搏动征。腹部平坦，无腹壁静脉显露，无胃肠型和蠕动波，腹部柔软，无压痛、反跳痛，肝脏未触及，脾脏未触及，未触及腹部包块，麦氏点无压痛及反跳痛，Murphy's征－，肾脏未触及，肝浊音界正常，肝肾区无明显肾区叩击痛，肝脾区无明显叩击痛，腹部叩诊鼓音，移动性浊音-，肠鸣音正常，无过水声，肛门及外生殖器未查，脊柱四肢见专科检查；双下肢无水肿；生理反射存在，病理反射未引出。
> 入院诊断：
>           中医诊断：
>                    心衰病
>                        气虚血瘀证
>           西医诊断：
>                   1.慢性心力衰竭急性失代偿
>                          心功能II级 NYHA分级
>                   2.病毒性扩张型心肌病
>                   3.肾病综合征
> 诊疗经过：入院后予内科护理常规，Ⅱ级护理，留陪人，低盐低脂低糖饮食，测血压，心电图，静脉血栓栓塞风险评估。完善血常规、尿常规检查、大便常规+潜血、尿蛋白/尿肌酐检测、电解质+生化、凝血四项+血浆D-二聚体、血清肌钙蛋白T测定、脑钠素-proBNP、缺血修饰白蛋白测定、颈动脉彩超、十二导联动态心电图等相关检验检查以协助诊疗。治疗上予左西孟旦注射液改善心功能，地高辛强心，可兰特控制心室率，美卓乐抗炎，注射用红花黄色素(乐坦)活血化瘀通络，极化液营养心肌、改善心肌代谢，诺新妥改善心功能，美达信调脂稳定斑块，百令胶囊补肾益精，骨化三醇、钙尔奇补钙等治疗。穴位贴敷辅助治疗，中药益气利水活血。患者Padua风险评估为1分，风险为低危，给予基本预防并做好病情告知工作，避免下肢深静脉穿刺或输液，指导患者做肢体关节活动并注意保暖，指导患者进行深呼吸促进血液回流，定期复查D二聚体、下肢静脉彩超等相关检验检查，结合宣传材料讲解下肢静脉血栓风险，鼓励患者改正不良生活习惯，避免久站久卧，鼓励患者多吃新鲜水果蔬菜并保持大小便通畅等。
> 出院诊断：
>           中医诊断：
>                  心衰病  
>                     气虚血瘀证
>           西医诊断：
>                  1.慢性心力衰竭急性失代偿
>                         心功能II级 NYHA分级
>                  2.病毒性扩张型心肌病
>                  3.肾病综合征 
>                  4.高尿酸血症
> 出院情况：患者无明显胸闷憋气、心悸，无气短、乏力，无有恶心、腹胀，无发热，无咳嗽咳痰，无腹痛，无头晕耳鸣，无眼干，纳眠可，二便调。查体：体温：36.2℃  脉搏：62次/分  呼吸：17次/分  血压：106/68mmHg Padua评分：1分（低危）。查体：患者青年，男,发育正常，营养良好，神志清楚，步入病房,查体合作，全身皮肤及粘膜无黄染,未见皮下出血,浅表淋巴结未及肿大。头颅五官无畸形，眼睑无水肿,巩膜无黄染,双侧瞳孔等大等圆，对光反射灵敏，外耳道无异常分泌物，鼻外观无畸形，口唇红润，伸舌居中，双侧扁桃体正常，表面未见脓性分泌物，颈软，无抵抗感，双侧颈静脉正常，气管居中，甲状腺未及肿大，未闻及血管杂音。胸廓正常,双肺呼吸音清晰，未闻及干、湿啰音，未闻及胸膜摩擦音。心界┐螅穆 2次/分，心律齐整，心音可,各瓣膜听诊区未闻及杂音，未闻及心包摩擦音。脉搏规整，无水冲脉、枪击音、毛细血管搏动征。腹部平坦，无腹壁静脉显露，无胃肠型和蠕动波，腹部柔软，无压痛、反跳痛，肝脏未触及，脾脏未触及，未触及腹部包块，麦氏点无压痛及反跳痛，Murphy's征－，肾脏未触及，肝浊音界正常，肝肾区无明显肾区叩击痛，肝脾区无明显叩击痛，腹部叩诊鼓音，移动性浊音-，肠鸣音正常，无过水声，肛门及外生殖器未查，脊柱四肢见专科检查；双下肢无水肿；生理反射存在，病理反射未引出。神经专科查体：神志清，精神可,吞咽正常左上肢近端肌力5级，远端肌力5级，右上肢近端肌力5级，远端肌力5级，左下肢近端肌力5级，远端肌力5级，右下肢近端肌力5级，远端肌力5级，双侧Hoffmann征阴性,双侧Babinski征阴性,双侧Oppenheim征阴性,双侧Gordon征阴性；共济运动：指鼻试验稳准,轮替试验稳准,跟膝胫试验阴性,闭目难立征阴性；脑膜刺激征阴性。
> 出院医嘱： 
>     1.避风寒、畅情志、适劳逸、节饮食；
>     2.定期门诊随诊；
>     3.出院后康复方案：（1）建议食用低盐低胆固醇、低脂肪，富含维生素、纤维素、优质蛋白的食物，建议每天盐摄入量控制在6g以下，限制饮酒，戒烟。（2）运动锻炼采取有氧运动方式，主要选择：慢跑、散步、打太极拳等，注意运动量适宜。每周至少三次，每次30-60分钟；
>     4.患者VTE风险评估评分为1分，风险为低危，无出血禁忌症及机械预防禁忌症，采取基本预防措施，向患者及家属介绍VTE相关知识及风险，给予患者健康宣教，嘱适量饮水，多食蔬果，适当运动，调节心情，降低血液粘稠度，指导患者做肢体关节运动，深呼吸运动，按摩腓肠肌等，促进血液回流。
>      5.出院随访方案：每日监测血压、心率并记录，门诊时携带，监测血糖，如有不适及时复查BNP、肌钙蛋白、血常规、门诊生化等，根据相关指标调整治疗方案；
>      6.甲泼尼龙片(美卓乐)(20mgP.O QD) 、盐酸伊伐布雷定片(可兰特)(5mgP.O BID) 、地高辛片(0.1250mgP.O QD) 、阿托伐他汀钙片(美达信)(20mgP.O QN) 、沙库巴曲缬沙坦钠片(诺欣妥)(25mgP.O BID) 、碳酸钙D3片(钙尔奇D片)(0.60gP.O QD) 、骨化三醇软胶囊(盖三淳)(0.25ugP.O QD) 、百令胶囊(2gP.O TID) 、琥珀酸美托洛尔缓释片(倍他乐克)(11.8750mgP.O QD)，中药继服。
>
> ​														医师签字：XXX

## Enhancement Prompt Design

### Diagnosis Enhancement

We introduce the prompt design of the corresponder and checker in the diagnosis enhancement module. The answer output by the Corresponder may not be what we expect, so we need a Checker to check the generated result. 

The prompt of the corresponder is shown below:

> As a skilled clinical coding professional, it is your responsibility to cluster procedure ICD-9 codes and laboratory test results to the appropriate diagnosis ICD-9 codes. Here's what you need to be aware of:
> 1. Each diagnosis code corresponds to a category.
> 2. The principle of clustering is “is it associated with the same disease” to ensure that the procedure is used to treat the diagnosed disease and that the laboratory test result is related to the diagnosed disease.
> 3. Each procedure code must and can only correspond to one diagnosis code.
> 4. Each diagnosis code must appear in the results.
> 5. The same laboratory test result may be categorized to multiple different diagnosis codes. 
> 6. The number of diagnosis codes and procedure codes output must be equal to the number of codes I gave you.
> 7. There may be some evidence in some chapters of the clinical record for your reference.
>
> Below, I'll provide you with the code to cluster and some sections from the clinical note:
> {len(diag)} diagnosis codes corresponding to the number of categories in the results: {diag}
> {len(proc)} procedure codes that must be classified only once: {proc}
> {len(test)} laboratory test results that can be categorized into multiple categories: {test}
> Chief Complaint: {chief}
> History of Present Illness: {history}
> Physical Exam: {exam}
>
> The format of the answer you give should be a list with the first level representing the clustered results for the different diagnoses, the first item on the second level being the diagnosis code for that cluster, the second item on the second level being the list of procedure codes for that cluster, and the third item on the second level being the list of laboratory test results. Do not output anything else. Please return the list without code blocks in plain text format. Example:
> '''
> [["E915", ["45.13"], []],
> ["038.8", ["00.14"], ["White Blood Cells: 30.5 K/uL"]],
> ["250.00", [], ["Glucose: 200mg/dL"]]]
> '''

If the number of lists is not prompted, the LLM response will have a high probability of missing information. The quality of LLM responses would be improved if a detailed example that encompasses many situations could be used. Due to funding constraints, we will only use simple examples here.

The prompt of the checker is shown below:

> The following is a list of multiple triples, each in the form of "[diagnosis, procedure list, laboratory event list]", where diagnosis and procedure are represented by ICD-9 codes:
> {enhanced_diagnoses_list}
> ﻿
> You can only answer "Yes" or "No". You output "Yes" only if the following two conditions are met, otherwise output "No", please return it in plain text.
> 1. The triplets in the list conform to medical logic. In each triplet, procedures and laboratory events may be related to the corresponding diagnosis (procedures are used to treat diagnoses, and laboratory events reflect the diagnosed condition).
> 2. The list given to you conforms to the python list format.

### Medication Mapping

Due to labor cost reasons, we use the labels generated by LLM in JustMed for subtask training. In order to ensure the correctness of the labels, we also designed a checker to assist in checking whether the generated r esults conform to medical logic and check the generated text format. 

The prompt of the corresponder is shown below:

> As a skilled clinical doctor, it is your responsibility to determine which diagnoses and procedures are being treated with each medication in a given prescription. Here are the things you need to pay attention to:
> 1. The diagnosis and treatment procedures are provided to you in the form of an ICD9 coding list.
> 2. The drugs in the prescription must be prescribed for at least one disease or diagnosis, therefore, you must treat at least one diagnosis or procedure for each drug.
> 3. The Brief Hospital Course chapter in the clinical notes may contain some facts about targeted medication, which you can refer to.
>
> Below I will provide you with a list of diagnoses, procedures, and medications, as well as the Brief Hospital Course section of the clinical notes for this prescription:
> {len(diag)} Diagnosis list: {diag}
> {len(proc)} Procedure list: {proc}
> {len(med)} Medication list: {med}
> Brief Hospital Course: {hosp_course}
>
> The result you give should be a list. Where the number of sublists is equal to the number of drugs, and each sublist is a correspondence between a drug and a diagnosis or procedure. You should not output any other text unrelated to the list. Please return the list without code blocks in plain text format. Return result example : [['drug1', ['diagnosis1'], ['procedure1']], ['drug2', ['diagnosis1', 'diagnosis2'], []]]

And the prompt of the checker is as follows:

> The following is a list of multiple triples, each of which is in the form of "[drug, diagnosis list, procedure list]", where diagnosis and procedure are represented by ICD-9 codes:
> {enhanced_answer_list}
> ﻿
> You can only answer "Yes" or "No". You output "Yes" only if the following two conditions are met, otherwise output "No", please return it in plain text.
> 1. The triples in the list conform to medical logic. In each triple, the drug can play the role of treatment procedure and diagnosis.
> 2. The list given to you conforms to the python list format.

## Explanation Generation

We introduce two prompts for LLM explanation generation. One is for the JustMed framework, as shown below:

> Below is a patient’s clinical record, divided by individual diagnoses. Each diagnosis is accompanied by its relevant procedures, lab test results, and recommended medications. Please evaluate the appropriateness of the drug recommendations, taking into account the specific details of each part, as well as the patient’s overall medical condition, including the combination of all diagnoses, procedures, and lab tests. Complete the following tasks.
> Patient Clinical Record:
>     Part 1:
>         Diagnosis: Diverticulitis of colon with hemorrhage
>         Procedure: Other partial excision of large intestine
>         Lab Test: None
>         Medications: Morphine sulfate, Cefazolin, Metronidazole, Sodium phosphate, Ondansetron
>     Part 2:
>         Diagnosis: Acute posthemorrhagic anemia
>         Procedure: Transfusion of packed cells
>         Lab Tests: Hematocrit: 35.5%, Hemoglobin: 11.4 g/dL, …
>         Medications: Ferrous sulfate
>     Part 3:
>         Diagnosis: Acidosis
>         Procedure: Venous catheterization, not elsewhere classified
>         Lab Tests: Bicarbonate: 28.0 mEq/L, Urea Nitrogen: 21.0 mg/dL
>         Medications: Magnesium sulfate, Potassium chloride, Ondansetron, Calcium gluconate, Potassium chloride
>     Part 4:
>         Diagnosis: Embolism and thrombosis of iliac artery
>         Procedure: Arteriography of other intra-abdominal arteries
>         Lab Test: None
>         Medications: None
>     Part 5:
>         Diagnosis: Type II diabetes mellitus
>         Procedure: None
>         Lab Test: Glucose: 147.0 mg/dL
>         Medications: Glyburide, …
>     Part 6:
>         Diagnosis: Unspecified essential hypertension
>         Procedure: None
>         Lab Test: None
>         Medications: Metoprolol, Prazosin HCl, Moexipril HCl, Furosemide
>     Part 7:
>         Diagnosis: Hypotension, unspecified
>         Procedure: None
>         Lab Test: Potassium: 3.2 mEq/L
>         Medications: Furosemide
> [Task]
>
> 1. For each part, are the recommended drugs appropriate given the diagnosis, procedure, and lab values?
> 2. Provide a reason for each drug's relevance or any concerns about its use.

The other is common to other medication recommendation models, as shown below:

> Below is a patient’s clinical record, including multiple diagnoses, procedures, lab test results, and a list of recommended medications. Please evaluate the appropriateness of the drug recommendations, considering the patient’s overall medical condition, including the combination of all diagnoses, procedures, and lab tests. Complete the following tasks.
> Patient Clinical Record:
> [Diagnoses]
> 	{Diagnoses}
> [Procedures]
> 	{Procedures}
> [Lab test]
> 	{Lab test}
> [Recommended Drugs]
> 	{Recommended Drugs}
> [Task]
> 1. Are the recommended drugs appropriate for the patient's diagnoses, procedures, and lab values?
> 2. Provide an explanation for each drug's relevance or potential issues.

The only difference between them is the data format.

In addition, we also published the results generated by the two prompts on **GPT4o**. Since the results generated by the large model are not stable, we only conduct some simple analyses. The results output by JustMed are more effective overall. Although it is slightly more complex and less concise, it is able to link drugs to specific diagnoses, procedures, and lab events, making it more insightful. LLM with the input of this structure also raised some important issues, such as the lack of anticoagulant therapy and the inappropriate use of certain drugs, which are ignored by the other explanation.

The result generated by GPT4o:

> Let's evaluate the appropriateness of the drug recommendations for each part of the patient’s clinical record based on the diagnosis, procedure, lab results, and the patient's overall medical condition.
> ### Part 1:
> **Diagnosis**: Diverticulitis of colon with hemorrhage  
> **Procedure**: Other partial excision of large intestine  
> **Lab Test**: None  
> **Medications**: Morphine sulfate, Cefazolin, Metronidazole, Sodium phosphate, Ondansetron  
> #### Evaluation:
> 1. **Morphine sulfate**: Appropriate for pain management after surgery, especially following a partial excision of the large intestine.
> 2. **Cefazolin**: Appropriate for surgical prophylaxis against infections, particularly in gastrointestinal surgeries where bacterial exposure is a concern.
> 3. **Metronidazole**: Appropriate for treating anaerobic infections, which are common in diverticulitis. It is often used in combination with other antibiotics like cefazolin.
> 4. **Sodium phosphate**: Typically used as a bowel cleanser pre-surgery or for treating constipation, but its use here is unclear unless it was used as part of bowel preparation prior to surgery.
> 5. **Ondansetron**: Appropriate for preventing or treating postoperative nausea and vomiting, commonly used in post-surgical care.
> **Concerns**: No significant concerns. All medications align well with the diagnosis and procedure.
> ---
> ### Part 2:
> **Diagnosis**: Acute posthemorrhagic anemia  
> **Procedure**: Transfusion of packed cells  
> **Lab Tests**: Hematocrit: 35.5%, Hemoglobin: 11.4 g/dL  
> **Medications**: Ferrous sulfate  
> #### Evaluation:
> 1. **Ferrous sulfate**: Appropriate for replenishing iron stores after blood loss due to acute hemorrhage. Although the hemoglobin and hematocrit levels are not severely low, the iron supplementation is beneficial for recovery.
> **Concerns**: Ferrous sulfate is appropriate, but follow-up lab tests to monitor iron levels and hemoglobin should be considered to adjust treatment as needed.
> ---
> ### Part 3:
> **Diagnosis**: Acidosis  
> **Procedure**: Venous catheterization, not elsewhere classified  
> **Lab Tests**: Bicarbonate: 28.0 mEq/L (normal), Urea Nitrogen: 21.0 mg/dL (normal)  
> **Medications**: Magnesium sulfate, Potassium chloride, Ondansetron, Calcium gluconate, Potassium chloride  
> #### Evaluation:
> 1. **Magnesium sulfate**: Appropriate if the patient has low magnesium levels or to prevent arrhythmias, though no specific magnesium lab values are provided. Its use should be based on electrolyte balance.
> 2. **Potassium chloride** (listed twice): Appropriate if the patient has hypokalemia (low potassium), but the lab value for potassium is not provided here. Excessive potassium supplementation without confirmed hypokalemia could pose risks of hyperkalemia.
> 3. **Ondansetron**: Appropriate for treating nausea, which could be related to metabolic imbalances.
> 4. **Calcium gluconate**: Typically used to stabilize cardiac function in the presence of hyperkalemia, but without lab values indicating hyperkalemia, its necessity is unclear.
> **Concerns**: The bicarbonate level is normal (28.0 mEq/L), which does not suggest acidosis, and the use of these medications, especially calcium gluconate and potassium chloride, should be based on confirmed electrolyte imbalances.
> ---
> ### Part 4:
> **Diagnosis**: Embolism and thrombosis of iliac artery  
> **Procedure**: Arteriography of other intra-abdominal arteries  
> **Lab Test**: None  
> **Medications**: None  
> #### Evaluation:
> 1. **No medications**: For embolism and thrombosis, anticoagulants (e.g., heparin, warfarin) are typically indicated to prevent clot propagation and recurrence. The lack of medication in this case raises concerns. The patient should likely be on anticoagulation therapy unless contraindicated.
> **Concerns**: The absence of anticoagulant therapy is concerning and should be addressed, as it increases the risk of further thromboembolic events.
> ---
> ### Part 5:
> **Diagnosis**: Type II diabetes mellitus  
> **Procedure**: None  
> **Lab Test**: Glucose: 147.0 mg/dL (elevated)  
> **Medications**: Glyburide  
> #### Evaluation:
> 1. **Glyburide**: Appropriate for managing blood glucose levels in Type II diabetes. However, glyburide is known for causing hypoglycemia, particularly in elderly patients or those with renal impairment.
> **Concerns**: Monitoring for hypoglycemia is necessary, and if the patient has kidney issues or fluctuating glucose levels, other agents like metformin or a DPP-4 inhibitor might be safer.
> ---
> ### Part 6:
> **Diagnosis**: Unspecified essential hypertension  
> **Procedure**: None  
> **Lab Test**: None  
> **Medications**: Metoprolol, Prazosin HCl, Moexipril HCl, Furosemide  
> #### Evaluation:
> 1. **Metoprolol**: Appropriate beta-blocker for hypertension, especially if there are concerns about heart rate control or cardiac conditions.
> 2. **Prazosin HCl**: Appropriate alpha-blocker, though typically reserved for resistant hypertension or specific cases like PTSD or pheochromocytoma.
> 3. **Moexipril HCl**: An ACE inhibitor, appropriate for hypertension management, especially if there are indications of renal involvement or heart failure.
> 4. **Furosemide**: Loop diuretic, often used in hypertension with fluid overload (e.g., heart failure). However, overuse could cause dehydration or worsen hypotension.
> **Concerns**: The combination of multiple antihypertensive agents, especially furosemide, raises the risk of hypotension, especially if not carefully monitored. Given the patient also has a diagnosis of **hypotension** (Part 7), careful consideration of the dosage and timing of these medications is needed.
> ---
> ### Part 7:
> **Diagnosis**: Hypotension, unspecified  
> **Procedure**: None  
> **Lab Test**: Potassium: 3.2 mEq/L (low)  
> **Medications**: Furosemide  
> #### Evaluation:
> 1. **Furosemide**: This is concerning. Furosemide is a diuretic that can cause or worsen hypotension and further deplete potassium, which is already low. Its use in a patient with hypotension and low potassium is inappropriate unless there is an overriding condition like fluid overload requiring diuresis.
> **Concerns**: Furosemide should be discontinued or used with extreme caution, and the potassium deficiency should be corrected before considering any diuretic therapy.
> ---
> ### Overall Considerations:
> - **Polypharmacy risks**: The patient is on multiple medications that could interact, especially given the mix of antihypertensives and diuretics. Monitoring for electrolyte imbalances (hypokalemia, hyperkalemia) and blood pressure swings (hypotension vs. hypertension) is essential.
> - **Lack of anticoagulation**: For the iliac artery thrombosis, anticoagulation should be considered unless contraindicated.
> - **Glyburide caution**: This medication requires careful monitoring due to its risk of hypoglycemia, especially in vulnerable patients.
> - **Furosemide risks**: Its use in a hypotensive patient with low potassium levels raises significant concerns, and dosage adjustments or discontinuation should be considered.
> In summary, while many of the medications are appropriate, certain medications like **furosemide** in the setting of hypotension and **the absence of anticoagulation** for arterial thrombosis need reevaluation. Monitoring for drug interactions and adjusting dosages based on the patient's current lab values and overall condition is crucial.

The result generated by GPT4o :

> ### Evaluation of Recommended Drugs
> 1. **Morphine Sulfate**:
>    - **Relevance**: Morphine is an opioid analgesic used for severe pain relief, likely relevant for postoperative pain management after the excision of the large intestine or in managing embolism-related discomfort.
>    - **Potential Issues**: In a patient with hypotension or diverticulitis, opioid use can exacerbate low blood pressure or cause constipation, which could complicate the patient's diverticulitis.
> 2. **Cefazolin**:
>    - **Relevance**: Cefazolin is a cephalosporin antibiotic often used perioperatively to prevent infections, especially in abdominal surgeries like partial colon excision.
>    - **Potential Issues**: Generally appropriate, though if the patient has a known allergy to beta-lactams, an alternative would be necessary.
> 3. **Metronidazole**:
>    - **Relevance**: Metronidazole is commonly used to treat anaerobic bacterial infections, which is crucial in abdominal infections or diverticulitis.
>    - **Potential Issues**: No major issues, but it’s important to monitor for gastrointestinal side effects, especially given the patient’s history of diverticulitis.
> 4. **Sodium Phosphate**:
>    - **Relevance**: Sodium phosphate can be used to manage hypophosphatemia, but it’s not clear from the lab results if phosphate levels were measured or indicated to be low.
>    - **Potential Issues**: In a patient with acidosis, sodium phosphate may not be ideal since phosphate metabolism can affect acid-base balance. Monitoring would be required.
> 5. **Ondansetron**:
>    - **Relevance**: Ondansetron is an antiemetic, useful for preventing nausea and vomiting post-surgery, particularly after general anesthesia or opioid use.
>    - **Potential Issues**: Safe and relevant in this context. No major concerns.
> 6. **Ferrous Sulfate**:
>    - **Relevance**: Ferrous sulfate is used to treat anemia, particularly iron-deficiency anemia. The patient’s acute posthemorrhagic anemia would benefit from iron supplementation.
>    - **Potential Issues**: This drug is appropriate, but gastrointestinal side effects, such as constipation, should be monitored, especially in a patient with diverticulitis.
> 7. **Magnesium Sulfate**:
>    - **Relevance**: Magnesium sulfate may be used to correct hypomagnesemia. However, there is no indication of low magnesium from the lab results.
>    - **Potential Issues**: Administering magnesium sulfate without confirming a deficiency could cause hypermagnesemia, leading to complications like hypotension, which this patient already suffers from.
> 8. **Potassium Chloride**:
>    - **Relevance**: The patient's potassium level (3.2 mEq/L) is low (normal range 3.5-5.0 mEq/L), so potassium supplementation is appropriate to correct hypokalemia.
>    - **Potential Issues**: This is an essential correction given the hypokalemia, but potassium supplementation should be carefully monitored to avoid overcorrection.
> 9. **Calcium Gluconate**:
>    - **Relevance**: Calcium gluconate is often used to correct hypocalcemia or to stabilize cardiac function in cases of hyperkalemia. No evidence from the labs suggests this is necessary.
>    - **Potential Issues**: Unnecessary supplementation of calcium may lead to hypercalcemia, especially if not indicated.
> 10. **Glyburide**:
>     - **Relevance**: Glyburide is an oral hypoglycemic agent used to manage type II diabetes mellitus. The patient's glucose level (147 mg/dL) indicates some hyperglycemia, making glyburide appropriate.
>     - **Potential Issues**: This drug increases insulin secretion and may cause hypoglycemia. It should be used cautiously, especially considering the patient’s other conditions like hypotension.
> 11. **Metoprolol**:
>     - **Relevance**: Metoprolol is a beta-blocker that is useful for treating hypertension. It also has benefits for cardiac protection in patients with embolism and thrombosis.
>     - **Potential Issues**: Since the patient also has hypotension, metoprolol may worsen this condition. Dose adjustments or alternative antihypertensive agents should be considered.
> 12. **Prazosin HCl**:
>     - **Relevance**: Prazosin is an alpha-blocker used for hypertension and potentially for managing symptoms of embolism.
>     - **Potential Issues**: Like metoprolol, prazosin can worsen hypotension, which is already a concern for this patient. Careful blood pressure monitoring is needed.
> 13. **Moexipril HCl**:
>     - **Relevance**: Moexipril is an ACE inhibitor used to treat hypertension, and it may benefit patients with diabetic nephropathy.
>     - **Potential Issues**: ACE inhibitors can cause hyperkalemia, which could be problematic if the patient is also receiving potassium chloride for hypokalemia. Close monitoring is required.
> 14. **Furosemide**:
>     - **Relevance**: Furosemide is a loop diuretic used to manage fluid overload or hypertension, especially in patients with heart or renal issues.
>     - **Potential Issues**: Furosemide can cause hypokalemia, which is already present in this patient. Combining it with potassium supplementation is common, but close monitoring of potassium levels is essential.
> ### Summary
> The recommended drugs largely align with the patient’s diagnoses and procedures, but some medications present potential risks, particularly in relation to the patient's existing hypotension and electrolyte imbalances. Close monitoring of electrolytes (potassium, calcium, magnesium) and blood pressure is critical due to the potential for drug interactions and the risks of exacerbating certain conditions.

## Acknowledgement

None
