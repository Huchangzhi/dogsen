"""
中国象棋图形界面
使用Pygame创建可视化棋盘
"""
import pygame
import numpy as np
import sys
from chinese_chess import ChineseChess


class ChineseChessRenderer:
    """中国象棋渲染器"""
    
    def __init__(self, board_width=9, board_height=10):
        self.board_width = board_width
        self.board_height = board_height
        self.square_size = 80  # 增加格子大小
        self.margin_x = 80     # 增加边距
        self.margin_y = 80
        self.width = self.board_width * self.square_size + 2 * self.margin_x
        self.height = self.board_height * self.square_size + 2 * self.margin_y
        
        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("中国象棋AI")
        self.clock = pygame.time.Clock()
        
        # 颜色定义
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BROWN = (222, 184, 135)  # 棋盘颜色
        self.WOOD = (160, 120, 60)    # 棋盘木色
        
        # 字体
        self.font = pygame.font.SysFont('SimHei', 32)  # 增大字体
        
        # 棋子名称映射
        self.piece_names = {
            1: '帅', 2: '仕', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
            -1: '将', -2: '士', -3: '象', -4: '马', -5: '车', -6: '炮', -7: '卒'
        }
    
    def draw_board(self, game_state):
        """绘制棋盘"""
        # 填充背景
        self.screen.fill(self.WOOD)
        
        # 绘制网格线
        # 水平线
        for i in range(self.board_height):
            start_x = self.margin_x
            end_x = self.margin_x + (self.board_width - 1) * self.square_size
            start_y = self.margin_y + i * self.square_size
            end_y = self.margin_y + i * self.square_size
            pygame.draw.line(self.screen, self.BLACK, (start_x, start_y), (end_x, end_y), 2)
        
        # 垂直线
        for j in range(self.board_width):
            start_x = self.margin_x + j * self.square_size
            end_x = self.margin_x + j * self.square_size
            start_y = self.margin_y
            end_y = self.margin_y + (self.board_height - 1) * self.square_size
            pygame.draw.line(self.screen, self.BLACK, (start_x, start_y), (end_x, end_y), 2)
        
        # 绘制特殊线（九宫格对角线）
        # 红方九宫
        x1, y1 = self.margin_x + 3 * self.square_size, self.margin_y + 7 * self.square_size
        x2, y2 = self.margin_x + 5 * self.square_size, self.margin_y + 9 * self.square_size
        pygame.draw.line(self.screen, self.BLACK, (x1, y1), (x2, y2), 2)  # 斜线
        pygame.draw.line(self.screen, self.BLACK, (x2, y1), (x1, y2), 2)  # 斜线
        
        # 黑方九宫
        x1, y1 = self.margin_x + 3 * self.square_size, self.margin_y
        x2, y2 = self.margin_x + 5 * self.square_size, self.margin_y + 2 * self.square_size
        pygame.draw.line(self.screen, self.BLACK, (x1, y1), (x2, y2), 2)  # 斜线
        pygame.draw.line(self.screen, self.BLACK, (x2, y1), (x1, y2), 2)  # 斜线
        
        # 河界文字
        river_text = self.font.render("楚河", True, self.BLACK)
        self.screen.blit(river_text, (self.margin_x + 1 * self.square_size, self.margin_y + 4.5 * self.square_size))
        river_text = self.font.render("汉界", True, self.BLACK)
        self.screen.blit(river_text, (self.margin_x + 6 * self.square_size, self.margin_y + 4.5 * self.square_size))
        
        # 绘制棋子
        for i in range(self.board_height):
            for j in range(self.board_width):
                piece = game_state.board[i, j]
                if piece != 0:
                    # 棋子应该绘制在线的交点上
                    x = self.margin_x + j * self.square_size
                    y = self.margin_y + i * self.square_size
                    
                    # 绘制棋子圆圈，确保在交叉点上
                    pygame.draw.circle(self.screen, self.WOOD, (x, y), self.square_size//3)
                    pygame.draw.circle(self.screen, self.BLACK, (x, y), self.square_size//3, 2)
                    
                    # 绘制棋子文字
                    piece_name = self.piece_names[piece]
                    text_color = self.RED if piece > 0 else self.BLUE
                    text_surface = self.font.render(piece_name, True, text_color)
                    text_rect = text_surface.get_rect(center=(x, y))
                    self.screen.blit(text_surface, text_rect)
        
        # 显示游戏信息
        info_font = pygame.font.SysFont('SimHei', 20)
        info_text = f"当前玩家: {'红方' if game_state.current_player == 1 else '黑方'}"
        text_surface = info_font.render(info_text, True, self.BLACK)
        self.screen.blit(text_surface, (10, 10))
        
        if game_state.game_over:
            font_large = pygame.font.SysFont('SimHei', 48)
            if game_state.winner == 1:
                winner_text = "红方获胜!"
            elif game_state.winner == -1:
                winner_text = "黑方获胜!"
            else:
                winner_text = "平局!"
            
            text_surface = font_large.render(winner_text, True, (255, 0, 0))
            text_rect = text_surface.get_rect(center=(self.width//2, self.height//2))
            self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()
    
    def get_square_from_mouse(self, pos):
        """从鼠标位置获取棋盘坐标"""
        x, y = pos
        col = (x - self.margin_x) // self.square_size
        row = (y - self.margin_y) // self.square_size
        
        if 0 <= row < self.board_height and 0 <= col < self.board_width:
            return int(row), int(col)
        return None, None
    
    def run_game(self, game):
        """运行游戏"""
        selected_pos = None
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not game.game_over:
                    row, col = self.get_square_from_mouse(event.pos)
                    if row is not None and col is not None:
                        if selected_pos is None:
                            # 选择棋子
                            piece = game.board[row, col]
                            if piece != 0 and ((piece > 0 and game.current_player == 1) or (piece < 0 and game.current_player == -1)):
                                selected_pos = (row, col)
                        else:
                            # 移动棋子
                            from_row, from_col = selected_pos
                            success = game.make_move(from_row, from_col, row, col)
                            if success:
                                selected_pos = None  # 成功移动后取消选择
                            else:
                                # 如果移动无效，检查是否点击了另一个己方棋子
                                piece = game.board[row, col]
                                if piece != 0 and ((piece > 0 and game.current_player == 1) or (piece < 0 and game.current_player == -1)):
                                    selected_pos = (row, col)
                                else:
                                    selected_pos = None  # 取消选择
            
            self.draw_board(game)
            # 如果有选中的棋子，高亮显示
            if selected_pos:
                row, col = selected_pos
                x = self.margin_x + col * self.square_size
                y = self.margin_y + row * self.square_size
                pygame.draw.rect(self.screen, (255, 255, 0), (x, y, self.square_size, self.square_size), 3)
                pygame.display.flip()
            
            self.clock.tick(60)
        
        pygame.quit()


def main():
    """主函数"""
    print("启动中国象棋图形界面...")
    
    game = ChineseChess()
    renderer = ChineseChessRenderer()
    renderer.run_game(game)


if __name__ == "__main__":
    main()