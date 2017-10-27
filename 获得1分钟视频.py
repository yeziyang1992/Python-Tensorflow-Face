import cv2
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./video/{}.avi'.format(input('请输入英文姓名 : ')), fourcc, 20.0, (640, 480))
count = 1
while(cap.isOpened()):

    ret, frame = cap.read()
    out.write(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'time left :{} s'.format(int(1200-count)//20), (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    count = count+1
    if count > 1200:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()