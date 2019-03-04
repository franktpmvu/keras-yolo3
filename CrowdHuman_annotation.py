import json
from os import getcwd
data_name='CH_annotation_train.odgt'
list_sample = [json.loads(x.rstrip()) for x in open(data_name, 'r')]

def convert_annotation(data, list_file):
    for d in data["gtboxes"]:
        classes=["person"]
        #print(img_name)
        gtbox=d
        cls=gtbox["tag"]
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        #nowboxname=["fbox","vbox","hbox"]
        #nowboxname=["fbox","hbox"]
        nowboxname=["vbox","hbox"]
        for nbox in nowboxname:
            try:
                nowcls=nowboxname.index(nbox)
                nowbox=[gtbox[nbox][0],gtbox[nbox][1],gtbox[nbox][0]+gtbox[nbox][2],gtbox[nbox][1]+gtbox[nbox][3],int(nowcls)]
                list_file.write(" " + ",".join([str(a) for a in nowbox]))
            except:
                print("error")
wd = getcwd()
image_set="CrowdHuman"
classes=["person"]
list_file=open('%s_onlyPVH_train.txt'%(image_set), 'w')
for d in list_sample:
    img_name=d["ID"]
    list_file.write('%s/data/CrowdHuman/images/%s.jpg'%(wd,img_name))
    convert_annotation(d,list_file)
    list_file.write('\n')
list_file.close()
