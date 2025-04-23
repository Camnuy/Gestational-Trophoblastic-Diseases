import sys
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from pathology_app import PathologyApp

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 创建启动界面
    pixmap = QPixmap("res/startup_image.jpg")
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()

    # 确保启动界面显示一段时间，例如3秒
    import time
    time.sleep(3)

    # 启动主应用
    ex = PathologyApp()
    ex.showFullScreen()
    ex.show()
    
    # 关闭启动界面
    splash.finish(ex)
    
    sys.exit(app.exec_())
