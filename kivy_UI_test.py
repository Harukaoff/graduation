
import draw_lib
import fem_lib

import numpy as np, math, os, itertools, pandas as pd, copy
from numpy.linalg import solve
import itertools

import kivy, japanize_kivy
from kivy.graphics import Color, Ellipse, Line, Triangle
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.screenmanager import (NoTransition, SlideTransition, 
    CardTransition, SwapTransition, FadeTransition, 
    WipeTransition, FallOutTransition, RiseInTransition)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.pagelayout import PageLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.behaviors import ButtonBehavior
from kivy.properties import ObjectProperty
from kivy.uix.image import Image
from kivy.app import runTouchApp
from kivy.factory import Factory
from kivy.uix.label import Label
from kivy.graphics import Translate
from kivy.graphics import Mesh
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.core.window import Window

# Window.fullscreen = 'auto' # 全画面表示にする

import warnings
warnings.simplefilter('ignore')

#kivyファイル
Builder.load_string('''
#:kivy 1.11.1
<DrawWidget>: #フレームを入力するページ
    BoxLayout:
        canvas.before:
            Color:
                rgba: 1,1,1,1
            Rectangle:
                pos: self.pos
                size: self.size
        orientation: "vertical"
        BoxLayout:
            Button:
                text: "Reset"
                color: 0.6, 0, 0, 1
                background_color: 1, 0.5, 0.5, 0.2
                on_release:
                    root.show_reset_popup()
            Button:
                text: "Frame"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "draw"
            Button:
                text: "Load, Support"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "input"
            Button:
                text: "display figure"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "result"

        BoxLayout:
            size_hint_y: 7.5
            DrawImage:
                texture: self.texture_image

        BoxLayout:
            size_hint_y: 1.5
            Button:
                text: "undo"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05

<InputWidget>: #条件を入力するページ
    BoxLayout:
        canvas.before:
            Color:
                rgba: 1,1,1,1
            Rectangle:
                pos: self.pos
                size: self.size
        orientation: "vertical"
        BoxLayout:
            Button:
                text: "Reset"
                color: 0.6, 0, 0, 1
                background_color: 1, 0.5, 0.5, 0.2
                on_release:
                    root.show_reset_popup()
            Button:
                text: "Frame"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "draw"
            Button:
                text: "Load, Support"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "input"
            Button:
                text: "display figure"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "result"

        BoxLayout:
            size_hint_y: 7.5
            InputImage:
                texture: self.texture_image
                canvas.before:
                    Color:
                        rgba: 1,1,1,1
                    Rectangle:
                        pos: self.pos
                        size: self.size

        BoxLayout:
            size_hint_y: 1.5
            orientation: "vertical"
            BoxLayout:
                Button:
                    text: "aaa"
                    background_color: 0.98, 0.98, 0.98, 1
                    on_state:

            BoxLayout:
                Button:
                    text: "bbb"
                    background_color: 0.98, 0.98, 0.98, 1
                    on_state:

<InputoldWidget>: #条件を入力するページ old
    BoxLayout:
        canvas.before:
            Color:
                rgba: 1,1,1,1
            Rectangle:
                pos: self.pos
                size: self.size
        orientation: "vertical"
        BoxLayout:
            Button:
                text: "Reset"
                color: 0.6, 0, 0, 1
                background_color: 1, 0.5, 0.5, 0.2
                on_release:
                    root.show_reset_popup()
            Button:
                text: "Frame"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "draw"
            Button:
                text: "Load, Support"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "input"
            Button:
                text: "display figure"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "result"

        BoxLayout:
            size_hint_y: 7.5
            InputImage:
                texture: self.texture_image
                canvas.before:
                    Color:
                        rgba: 1,1,1,1
                    Rectangle:
                        pos: self.pos
                        size: self.size

        BoxLayout:
            size_hint_y: 1.5
            orientation: "vertical"
            BoxLayout:
                ToggleButton:
                    text: "ピン支点"
                    group: "toggle_1"
                    background_color: 0.98, 0.98, 0.98, 1
                    on_state:
                        root.toggle_1_state(self.text, self.state)
                    state: "down"
                ToggleButton:
                    text: "ピンローラー支点"
                    group: "toggle_1"
                    background_color: 0.98, 0.98, 0.98, 1
                    on_state:
                        root.toggle_1_state(self.text, self.state)
                ToggleButton:
                    text: "固定端"
                    group: "toggle_1"
                    background_color: 0.98, 0.98, 0.98, 1
                    on_state:
                        root.toggle_1_state(self.text, self.state)
                ToggleButton:
                    text: "ピン節点(未実装)"
                    group: "toggle_1"
                    background_color: 0.98, 0.98, 0.98, 1
                    on_state:
                        root.toggle_1_state(self.text, self.state)
            BoxLayout:
                ToggleButton:
                    text: "集中荷重"
                    group: "toggle_1"
                    background_color: 0.98, 0.98, 0.98, 1
                    on_state:
                        root.toggle_1_state(self.text, self.state)
                ToggleButton:
                    text: "分布荷重(未実装)"
                    group: "toggle_1"
                    background_color: 0.98, 0.98, 0.98, 1
                    on_state:
                        root.toggle_1_state(self.text, self.state)
                ToggleButton:
                    text: "モーメント荷重"
                    group: "toggle_1"
                    background_color: 0.98, 0.98, 0.98, 1
                    on_state:
                        root.toggle_1_state(self.text, self.state)
                ToggleButton:
                    text: "8"
                    group: "toggle_1"
                    background_color: 0.98, 0.98, 0.98, 1
                    on_state:
                        root.toggle_1_state(self.text, self.state)

<ResultWidget>: #結果を出力するページ
    BoxLayout:
        canvas.before:
            Color:
                rgba: 1,1,1,1
            Rectangle:
                pos: self.pos
                size: self.size
        orientation: "vertical"
        BoxLayout:
            Button:
                text: "Reset"
                color: 0.6, 0, 0, 1
                background_color: 1, 0.5, 0.5, 0.2
                on_release:
                    root.show_reset_popup()
            Button:
                text: "Frame"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "draw"
            Button:
                text: "Load, Support"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "input"
            Button:
                text: "display figure"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.manager.current = "result"

        BoxLayout:
            size_hint_y: 7.5
            canvas.before:
                Color:
                    rgba: 1,1,1,1
                Rectangle:
                    pos: self.pos
                    size: self.size

        BoxLayout:
            size_hint_y: 1.5
            Button:
                text: "Form"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.F_1()
            Button:
                text: "N"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.F_2()
            Button:
                text: "Q"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.F_3()
            Button:
                text: "M"
                color: 0, 0, 0, 1
                background_color: 0, 0, 0, 0.05
                on_release:
                    root.F_4()

<ResetPopup>: #リセット確認画面
    BoxLayout:
        orientation: "vertical"
        Label:
            size_hint_y: 7
            text: "delete this description"
            color: 1,1,1,1

        BoxLayout:
            Button:
                text: "OK"
                background_color: 0.4, 0.4, 0.4, 1
                on_release:
                    root.reset_do()
                    root.close_reset_popup()
            Button:
                text: "CANCEL"
                background_color: 0.4, 0.4, 0.4, 1
                on_release:
                    root.close_reset_popup()
''')



class DrawWidget(Screen): #drawの画面

    def __init__(self, **kwargs):
        super(DrawWidget, self).__init__(**kwargs)

    def on_enter(self): # 画面を表示した際に行う操作
        node_df = draw_lib.node_df
        line_df = draw_lib.line_df
        line_list = draw_lib.make_line(node_df, line_df)

        with self.canvas.after:
            Color(0, 0, 0)
            for n in line_list:
                line = Line(points=(n[0][0], n[0][1], n[1][0], n[1][1]), width=2)

    def update_draw(self): # on_enter()で描写した線分を消去する際に使用
        self.canvas.after.clear()

    # リセット
    def show_reset_popup(self):
        content = ResetPopup(popup_close=self.close_reset_popup, draw_widget=self)
        self.popup = Popup(title='Confirm', content=content, size_hint=(0.6, 0.6), auto_dismiss=True)
        content.draw_widget = self
        self.popup.open()

    def close_reset_popup(self):
        self.popup.dismiss()

class DrawImage(Image): #drowの描写エリア画面

    def __init__(self, **kwargs):
        super(DrawImage, self).__init__(**kwargs)
        self.texture_image = Texture.create(size=self.size)

    def on_touch_down(self, touch):
        p.clear()
        p.append([touch.x, touch.y])
        with self.canvas:
            Color(0, 0, 0)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=2)

    def on_touch_move(self, touch):
        p.append([touch.x, touch.y])
        touch.ud['line'].points += [touch.x, touch.y]

    def on_touch_up(self, touch):
        self.canvas.clear()
        if (Window.height / 20)*3 < touch.y < (Window.height / 20)*18: #タッチしたy座標が描写範囲内だった場合のみ線分処理を実行
            s = draw_lib.find_fs(p, 14) #特徴点抽出

            with self.canvas:
                Color(0, 0, 0)

                #入力した線分をnode_df,line_dfに追加するプログラム
                dfs = draw_lib.add_df(s)
                node_df, line_df = dfs[0], dfs[1]

                #node_dfとline_dfからline_listを生成するプログラム
                line_list = draw_lib.make_line(node_df, line_df)
                for n in line_list:
                    Line(points=(n[0][0], n[0][1], n[1][0], n[1][1]), width=2)
        
            backgrounds.append([node_df, line_df, input_df]) #履歴にdataframeを追加する

class InputWidget(Screen): #inputの画面

    def __init__(self, **kwargs):
        super(InputWidget, self).__init__(**kwargs)
        self.lines = []  # 描画した線分を保持するリスト

    def on_enter(self): # 画面を表示した際に行う操作
        node_df = draw_lib.node_df
        line_df = draw_lib.line_df
        line_list = draw_lib.make_line(node_df, line_df)

        with self.canvas:
            Color(0.2, 0.2, 0.2, 0.6)
            for n in line_list:
                line = Line(points=(n[0][0], n[0][1], n[1][0], n[1][1]), width=2.5)
                self.lines.append(line)

    def on_leave(self, *args): # 画面を閉じた際に行う操作, self.linesに格納した線分データのみを削除する
        for line in self.lines:
            self.canvas.remove(line)
        return super().on_leave(*args)

    # リセット
    def show_reset_popup(self):
        content = ResetPopup(popup_close=self.close_reset_popup, draw_widget=self)
        self.popup = Popup(title='Confirm', content=content, size_hint=(0.6, 0.6), auto_dismiss=True)
        content.draw_widget = self
        self.popup.open()

    def close_reset_popup(self):
        self.popup.dismiss()

class InputImage(Image): #inputの描写エリア画面

    def __init__(self, **kwargs):
        super(InputImage, self).__init__(**kwargs)
        self.texture_image = Texture.create(size=self.size)

    def on_touch_down(self, touch):
        self.canvas.clear()
        if (Window.height / 20)*3 < touch.y < (Window.height / 20)*18: #タッチしたy座標が描写範囲内だった場合のみ線分処理を実行
            input_df = draw_lib.input_condition([touch.x, touch.y], InputImage.toggle_1) #条件入力のdataframe作成
            draw_lines = draw_lib.draw_condition() #条件描写データ
            with self.canvas:
                Color(1, 0, 0)
                for n in draw_lines:
                    if n[0] == 'line':
                        Line(points=(sum(n[1:], [])), width=1.25)
                    if n[0] == 'circle':
                        Line(ellipse=(n[1][0] - n[2]/2, n[1][1] - n[2]/2, n[2], n[2], n[3], n[4]), width=1.25)
                    if n[0] == 'triangle':
                        Triangle(points=(sum(n[1:], [])), width=1.25)

            backgrounds.append([node_df, line_df, input_df]) #履歴にdataframeを追加する

    def on_touch_move(self, touch):
        pass

    def on_touch_up(self, touch):
        pass

class ResultWidget(Screen): #resultの画面

    def __init__(self, **kwargs):
        super(ResultWidget, self).__init__(**kwargs)
        self.lines = []  # 描画した線分を保持するリスト

    def on_enter(self): # 画面を表示した際に行う操作
        try: # 構造が入力されていない場合計算過程でエラーが発生するため例外処理
            # FEM計算実行
            elements_df, nodes_df = draw_lib.make_dfs()
            D_R, M_S = fem_lib.fem_calc(elements_df, nodes_df) # 変位・反力 / 応力・変形
            points_df_list = draw_lib.make_figure(M_S) # 座標dfリスト
            self.draw_list = draw_lib.draw_fig(points_df_list)

            # 構造ベース描写
            node_df = draw_lib.node_df
            line_df = draw_lib.line_df
            self.line_list = draw_lib.make_line(node_df, line_df)

            self.canvas.after.clear()
            with self.canvas.after:
                Color(0.2, 0.2, 0.2, 0.6)
                for n in self.line_list:
                    line = Line(points=(n[0][0], n[0][1], n[1][0], n[1][1]), width=1.5)
                    self.lines.append(line)

            # 初期値図として変形図を描写
            with self.canvas.after:
                Color(0.0, 0.0, 0.0, 1)
                for n in self.draw_list[0]:
                    line = Line(points=(n), width=2.0)
        except ValueError:
            pass

    def on_leave(self, *args):
        self.canvas.after.clear()

    def F_1(self): # 変形図作図
        self.canvas.after.clear()
        with self.canvas.after:
                Color(0.2, 0.2, 0.2, 0.6)
                for n in self.line_list:
                    line = Line(points=(n[0][0], n[0][1], n[1][0], n[1][1]), width=1.5)
                    self.lines.append(line)
        with self.canvas.after:
            Color(0.0, 0.0, 0.0, 1)
            for n in self.draw_list[0]:
                line = Line(points=(n), width=2.0)

    def F_2(self): # N図作図
        self.canvas.after.clear()
        with self.canvas.after:
                Color(0.0, 0.0, 0.0, 1)
                for n in self.line_list:
                    line = Line(points=(n[0][0], n[0][1], n[1][0], n[1][1]), width=1.5)
                    self.lines.append(line)
        with self.canvas.after:
            Color(0.1, 0.0, 1.0, 1)
            for n in self.draw_list[1]:
                line = Line(points=(n), width=0.9)

            Color(0.5, 0.5, 1.0, 1)
            for n in self.draw_list[2]:
                line = Line(points=(n), width=0.9)

    def F_3(self): # Q図作図
        self.canvas.after.clear()
        with self.canvas.after:
                Color(0.0, 0.0, 0.0, 1)
                for n in self.line_list:
                    line = Line(points=(n[0][0], n[0][1], n[1][0], n[1][1]), width=1.5)
                    self.lines.append(line)
        with self.canvas.after:
            Color(0.1, 0.0, 1.0, 1)
            for n in self.draw_list[3]:
                line = Line(points=(n), width=0.9)

            Color(0.5, 0.5, 1.0, 1)
            for n in self.draw_list[4]:
                line = Line(points=(n), width=0.9)

    def F_4(self): # M図作図
        self.canvas.after.clear()
        with self.canvas.after:
                Color(0.0, 0.0, 0.0, 1)
                for n in self.line_list:
                    line = Line(points=(n[0][0], n[0][1], n[1][0], n[1][1]), width=1.5)
                    self.lines.append(line)
        with self.canvas.after:
            Color(0.1, 0.0, 1.0, 1)
            for n in self.draw_list[5]:
                line = Line(points=(n), width=0.9)

            Color(0.5, 0.5, 1.0, 1)
            for n in self.draw_list[6]:
                line = Line(points=(n), width=0.9)

    # リセット
    def show_reset_popup(self):
        content = ResetPopup(popup_close=self.close_reset_popup, draw_widget=self)
        self.popup = Popup(title='Confirm', content=content, size_hint=(0.6, 0.6), auto_dismiss=True)
        content.draw_widget = self
        self.popup.open()

    def close_reset_popup(self):
        self.popup.dismiss()

class ResetPopup(Screen): #reset

    popup_close = ObjectProperty(None)
    draw_widget = ObjectProperty(None)

    def reset_do(self): #画面リセット時の挙動, データフレームのリセットを実行する
        node_df = pd.DataFrame(columns=['No', 'x', 'y'])
        line_df = pd.DataFrame(columns=['s_node', 'e_node'])
        input_df = pd.DataFrame(columns=['No','stick', 's_pos', 'load', 'l_pos'])
        draw_lib.node_df = node_df
        draw_lib.line_df = line_df
        draw_lib.input_df = input_df

        # 画面が'draw'の場合, 線分が残るためリセットを行う
        if self.draw_widget.manager.current == 'draw':
            self.draw_widget.update_draw()

        self.draw_widget.manager.current = 'draw'

    def close_reset_popup(self):
        self.popup_close()



class MyApp(App):

    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        self.title = 'app'

    def build(self):
        self.sm = ScreenManager(transition=NoTransition())
        self.sm.add_widget(DrawWidget(name='draw'))
        self.sm.add_widget(InputWidget(name='input'))
        self.sm.add_widget(ResultWidget(name='result'))
        self.sm.add_widget(ResetPopup(name='reset'))
        return self.sm




p = []

node_df = pd.DataFrame(columns=['No', 'x', 'y'])
line_df = pd.DataFrame(columns=['s_node', 'e_node'])
input_df = pd.DataFrame(columns=['No', 'stick', 's_pos', 'load', 'l_pos'])

draw_lib.node_df = node_df
draw_lib.line_df = line_df
draw_lib.input_df = input_df

backgrounds = [] #入力の履歴

MyApp().run()


