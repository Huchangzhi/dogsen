"""
中国象棋环境实现
包含棋盘表示、规则检查、移动逻辑等功能
"""
import numpy as np
from typing import List, Tuple, Optional


class ChineseChess:
    """
    中国象棋游戏环境
    棋盘大小为9x10
    """
    
    # 棋子定义
    EMPTY = 0
    RED_KING = 1
    RED_ADVISOR = 2
    RED_ELEPHANT = 3
    RED_HORSE = 4
    RED_CHARIOT = 5
    RED_CANNON = 6
    RED_PAWN = 7
    BLACK_KING = -1
    BLACK_ADVISOR = -2
    BLACK_ELEPHANT = -3
    BLACK_HORSE = -4
    BLACK_CHARIOT = -5
    BLACK_CANNON = -6
    BLACK_PAWN = -7
    
    PIECE_NAMES = {
        0: '.',
        1: 'K', 2: 'A', 3: 'E', 4: 'H', 5: 'R', 6: 'C', 7: 'P',
        -1: 'k', -2: 'a', -3: 'e', -4: 'h', -5: 'r', -6: 'c', -7: 'p'
    }
    
    def __init__(self):
        self.board = np.zeros((10, 9), dtype=np.int8)  # 10行9列
        self.current_player = 1  # 1: 红方, -1: 黑方
        self.game_over = False
        self.winner = None
        self.move_history = []  # 记录移动历史
        self.position_history = []  # 记录局面历史，用于检测循环
        self.repeat_limit = 3  # 重复局面限制
        self.setup_board()
    
    def setup_board(self):
        """设置初始棋盘布局"""
        # 红方（下方）
        self.board[9, 0] = self.RED_CHARIOT
        self.board[9, 1] = self.RED_HORSE
        self.board[9, 2] = self.RED_ELEPHANT
        self.board[9, 3] = self.RED_ADVISOR
        self.board[9, 4] = self.RED_KING
        self.board[9, 5] = self.RED_ADVISOR
        self.board[9, 6] = self.RED_ELEPHANT
        self.board[9, 7] = self.RED_HORSE
        self.board[9, 8] = self.RED_CHARIOT
        
        self.board[7, 1] = self.RED_CANNON
        self.board[7, 7] = self.RED_CANNON
        
        self.board[6, 0] = self.RED_PAWN
        self.board[6, 2] = self.RED_PAWN
        self.board[6, 4] = self.RED_PAWN
        self.board[6, 6] = self.RED_PAWN
        self.board[6, 8] = self.RED_PAWN
        
        # 黑方（上方）
        self.board[0, 0] = self.BLACK_CHARIOT
        self.board[0, 1] = self.BLACK_HORSE
        self.board[0, 2] = self.BLACK_ELEPHANT
        self.board[0, 3] = self.BLACK_ADVISOR
        self.board[0, 4] = self.BLACK_KING
        self.board[0, 5] = self.BLACK_ADVISOR
        self.board[0, 6] = self.BLACK_ELEPHANT
        self.board[0, 7] = self.BLACK_HORSE
        self.board[0, 8] = self.BLACK_CHARIOT
        
        self.board[2, 1] = self.BLACK_CANNON
        self.board[2, 7] = self.BLACK_CANNON
        
        self.board[3, 0] = self.BLACK_PAWN
        self.board[3, 2] = self.BLACK_PAWN
        self.board[3, 4] = self.BLACK_PAWN
        self.board[3, 6] = self.BLACK_PAWN
        self.board[3, 8] = self.BLACK_PAWN
    
    def reset(self):
        """重置游戏"""
        self.board = np.zeros((10, 9), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.setup_board()
    
    def is_valid_move(self, from_row, from_col, to_row, to_col) -> bool:
        """检查移动是否有效"""
        if self.game_over:
            return False
        
        # 检查边界
        if not (0 <= from_row < 10 and 0 <= from_col < 9 and 
                0 <= to_row < 10 and 0 <= to_col < 9):
            return False
        
        piece = self.board[from_row, from_col]
        
        # 检查是否是己方棋子
        if piece == 0 or (piece > 0 and self.current_player < 0) or (piece < 0 and self.current_player > 0):
            return False
        
        # 检查目标位置是否为己方棋子
        target_piece = self.board[to_row, to_col]
        if target_piece != 0 and ((piece > 0 and target_piece > 0) or (piece < 0 and target_piece < 0)):
            return False
        
        # 检查帅（将）是否照面
        if self._would_kings_face(piece, from_row, from_col, to_row, to_col):
            return False
        
        # 根据棋子类型检查移动规则
        piece_type = abs(piece)
        
        if piece_type == 1:  # 帅/将
            return self._is_valid_king_move(from_row, from_col, to_row, to_col)
        elif piece_type == 2:  # 士/仕
            return self._is_valid_advisor_move(from_row, from_col, to_row, to_col)
        elif piece_type == 3:  # 相/象
            return self._is_valid_elephant_move(from_row, from_col, to_row, to_col)
        elif piece_type == 4:  # 马/馬
            return self._is_valid_horse_move(from_row, from_col, to_row, to_col)
        elif piece_type == 5:  # 车/車
            return self._is_valid_chariot_move(from_row, from_col, to_row, to_col)
        elif piece_type == 6:  # 炮/砲
            return self._is_valid_cannon_move(from_row, from_col, to_row, to_col)
        elif piece_type == 7:  # 兵/卒
            return self._is_valid_pawn_move(from_row, from_col, to_row, to_col)
        
        return False
    
    def _would_kings_face(self, moving_piece, from_row, from_col, to_row, to_col):
        """检查移动后是否会形成帅（将）照面"""
        # 创建临时棋盘来模拟移动
        temp_board = self.board.copy()
        temp_board[to_row, to_col] = temp_board[from_row, from_col]
        temp_board[from_row, from_col] = 0
        
        # 查找两个王的位置
        red_king_pos = None
        black_king_pos = None
        for r in range(10):
            for c in range(9):
                if temp_board[r, c] == self.RED_KING:
                    red_king_pos = (r, c)
                elif temp_board[r, c] == self.BLACK_KING:
                    black_king_pos = (r, c)
        
        if red_king_pos and black_king_pos:
            red_row, red_col = red_king_pos
            black_row, black_col = black_king_pos
            # 如果两个王在同一列，且中间没有其他棋子，则照面
            if red_col == black_col:
                # 检查两王之间是否有棋子
                min_row, max_row = min(red_row, black_row), max(red_row, black_row)
                has_piece_between = False
                for r in range(min_row + 1, max_row):
                    if temp_board[r, red_col] != 0:
                        has_piece_between = True
                        break
                if not has_piece_between:
                    return True  # 会形成照面，不允许移动
        
        return False
    
    def _is_valid_king_move(self, from_row, from_col, to_row, to_col):
        """检查帅/将移动是否有效"""
        # 只能在九宫内移动，每次一步
        if self.current_player == 1:  # 红方
            if not (3 <= to_col <= 5 and 7 <= to_row <= 9):
                return False
        else:  # 黑方
            if not (3 <= to_col <= 5 and 0 <= to_row <= 2):
                return False
        
        # 检查是否只移动了一格
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        return (row_diff + col_diff == 1)
    
    def _is_valid_advisor_move(self, from_row, from_col, to_row, to_col):
        """检查士/仕移动是否有效"""
        # 只能在九宫内斜向移动一格
        if self.current_player == 1:  # 红方
            if not (3 <= to_col <= 5 and 7 <= to_row <= 9):
                return False
        else:  # 黑方
            if not (3 <= to_col <= 5 and 0 <= to_row <= 2):
                return False
        
        # 检查是否斜向移动一格
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        return (row_diff == 1 and col_diff == 1)
    
    def _is_valid_elephant_move(self, from_row, from_col, to_row, to_col):
        """检查相/象移动是否有效"""
        # 斜向移动两格，不能过河，且象眼不能有棋子
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        # 检查是否斜向移动两格
        if not (row_diff == 2 and col_diff == 2):
            return False
        
        # 检查是否过河
        if self.current_player == 1 and to_row < 5:  # 红方不能过河
            return False
        if self.current_player == -1 and to_row > 4:  # 黑方不能过河
            return False
        
        # 检查象眼是否有棋子
        eye_row = (from_row + to_row) // 2
        eye_col = (from_col + to_col) // 2
        
        return self.board[eye_row, eye_col] == 0
    
    def _is_valid_horse_move(self, from_row, from_col, to_row, to_col):
        """检查马/馬移动是否有效"""
        # L形移动，且马腿不能被蹩
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        # 检查是否L形移动
        if not ((row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)):
            return False
        
        # 检查马腿是否被蹩
        if row_diff == 2:  # 竖向移动
            leg_row = from_row + (1 if to_row > from_row else -1)
            leg_col = from_col
        else:  # 横向移动
            leg_row = from_row
            leg_col = from_col + (1 if to_col > from_col else -1)
        
        return self.board[leg_row, leg_col] == 0
    
    def _is_valid_chariot_move(self, from_row, from_col, to_row, to_col):
        """检查车/車移动是否有效"""
        # 横向或纵向移动，路径上不能有棋子
        if from_row != to_row and from_col != to_col:
            return False  # 不是横移也不是竖移
        
        # 检查路径上是否有棋子
        if from_row == to_row:  # 横向移动
            start_col = min(from_col, to_col) + 1
            end_col = max(from_col, to_col)
            for col in range(start_col, end_col):
                if self.board[from_row, col] != 0:
                    return False
        else:  # 纵向移动
            start_row = min(from_row, to_row) + 1
            end_row = max(from_row, to_row)
            for row in range(start_row, end_row):
                if self.board[row, from_col] != 0:
                    return False
        
        return True
    
    def _is_valid_cannon_move(self, from_row, from_col, to_row, to_col):
        """检查炮/砲移动是否有效"""
        if from_row != to_row and from_col != to_col:
            return False  # 不是横移也不是竖移
        
        # 计算路径上的棋子数量
        count = 0
        if from_row == to_row:  # 横向移动
            start_col = min(from_col, to_col) + 1
            end_col = max(from_col, to_col)
            for col in range(start_col, end_col):
                if self.board[from_row, col] != 0:
                    count += 1
        else:  # 纵向移动
            start_row = min(from_row, to_row) + 1
            end_row = max(from_row, to_row)
            for row in range(start_row, end_row):
                if self.board[row, from_col] != 0:
                    count += 1
        
        # 如果目标位置有棋子，则必须跳过一个棋子（炮打隔子）
        target_piece = self.board[to_row, to_col]
        if target_piece != 0:
            return count == 1
        else:
            # 如果目标位置为空，则路径上不能有棋子
            return count == 0
    
    def _is_valid_pawn_move(self, from_row, from_col, to_row, to_col):
        """检查兵/卒移动是否有效"""
        row_diff = to_row - from_row
        col_diff = abs(to_col - from_col)
        
        if self.current_player == 1:  # 红方
            # 红方兵过河前只能向前，过河后可向前或向左右
            if from_row >= 5:  # 未过河
                return (row_diff == -1 and col_diff == 0)
            else:  # 已过河
                return ((row_diff == -1 and col_diff == 0) or (row_diff == 0 and col_diff == 1))
        else:  # 黑方
            # 黑方兵过河前只能向前，过河后可向前或向左右
            if from_row <= 4:  # 未过河
                return (row_diff == 1 and col_diff == 0)
            else:  # 已过河
                return ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1))
    
    def make_move(self, from_row, from_col, to_row, to_col) -> bool:
        """执行移动"""
        if not self.is_valid_move(from_row, from_col, to_row, to_col):
            return False
        
        piece = self.board[from_row, from_col]
        captured_piece = self.board[to_row, to_col]
        
        # 执行移动
        self.board[to_row, to_col] = piece
        self.board[from_row, from_col] = 0
        
        # 记录移动
        move = ((from_row, from_col), (to_row, to_col))
        self.move_history.append(move)
        
        # 记录当前局面
        self.position_history.append(self.board.tobytes())
        
        # 检查是否将死对方
        if abs(captured_piece) == 1:  # 捉到对方将/帅
            self.game_over = True
            self.winner = self.current_player
        # 检查是否出现重复局面（长将、长捉等循环）
        elif self._check_repetition():
            self.game_over = True
            self.winner = 0  # 平局
        
        # 切换玩家
        self.current_player = -self.current_player
        
        return True
    
    def _check_repetition(self) -> bool:
        """检查是否出现重复局面（长将、长捉等循环）"""
        if len(self.position_history) < 20:  # 提高最低检查步数，从16改为20
            return False

        # 检查最近的若干个局面是否重复出现
        recent_positions = self.position_history[-15:]  # 检查最近15个局面，从12改为15
        unique_positions = set(recent_positions)

        # 如果最近15个局面中，独特局面少于一定数量，认为是循环
        if len(unique_positions) < 2:  # 15个局面中少于2个独特局面才认为是循环，从3改为2
            return True

        # 检查是否出现完全相同局面超过限制次数
        current_position = self.position_history[-1]
        # 优化：只检查最近的几次，而不是全部历史
        recent_check = self.position_history[-30:]  # 只检查最近30个局面，从24改为30
        count = 0
        for pos in recent_check:
            if pos == current_position:
                count += 1
                if count >= self.repeat_limit:
                    return True

        return False
    
    def get_winner(self):
        """获取获胜者"""
        return self.winner
    
    def get_valid_moves(self, row, col) -> List[Tuple[int, int]]:
        """获取指定位置棋子的所有有效移动"""
        valid_moves = []
        for r in range(10):
            for c in range(9):
                if self.is_valid_move(row, col, r, c):
                    valid_moves.append((r, c))
        return valid_moves
    
    def get_all_valid_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """获取当前玩家所有有效移动"""
        valid_moves = []
        for r in range(10):
            for c in range(9):
                piece = self.board[r, c]
                if piece != 0 and ((piece > 0 and self.current_player > 0) or (piece < 0 and self.current_player < 0)):
                    for tr in range(10):
                        for tc in range(9):
                            if self.is_valid_move(r, c, tr, tc):
                                valid_moves.append(((r, c), (tr, tc)))
        return valid_moves
    
    def print_board(self):
        """打印当前棋盘状态"""
        print("\n当前棋盘状态:")
        print("  ", end="")
        for i in range(9):
            print(f"{i:2}", end=" ")
        print()
        
        for i in range(10):
            print(f"{i:2}", end=" ")
            for j in range(9):
                piece = self.board[i, j]
                print(f"{self.PIECE_NAMES[piece]:2}", end=" ")
            print()
        
        print(f"\n当前玩家: {'红方' if self.current_player == 1 else '黑方'}")
        
        if self.game_over:
            if self.winner == 1:
                print("游戏结束: 红方获胜!")
            elif self.winner == -1:
                print("游戏结束: 黑方获胜!")
    
    def get_state(self) -> np.ndarray:
        """获取当前状态用于神经网络输入
        返回形状为 (10, 9, 15) 的张量
        每个位置用15个平面表示（7种棋子×2方 + 1个当前玩家标识）
        """
        state = np.zeros((10, 9, 15), dtype=np.float32)
        
        for r in range(10):
            for c in range(9):
                piece = self.board[r, c]
                if piece != 0:
                    piece_type = abs(piece) - 1  # 0-6 对应 7种棋子
                    if piece > 0:  # 红方
                        state[r, c, piece_type] = 1.0
                    else:  # 黑方
                        state[r, c, piece_type + 7] = 1.0
        
        # 添加当前玩家标识平面 (第15个平面)
        state[:, :, 14] = 1.0 if self.current_player == 1 else -1.0
        
        return state


def test_chinese_chess():
    """测试中国象棋环境"""
    print("=== 测试中国象棋环境 ===")
    game = ChineseChess()
    game.print_board()
    
    # 尝试一些移动
    print("\n尝试移动红方炮到中间位置...")
    success = game.make_move(7, 1, 5, 1)
    if success:
        print("移动成功!")
        game.print_board()
    else:
        print("移动失败!")
    
    if not game.game_over:
        print("\n尝试移动黑方炮应对...")
        success = game.make_move(2, 1, 4, 1)
        if success:
            print("移动成功!")
            game.print_board()
        else:
            print("移动失败!")


if __name__ == "__main__":
    test_chinese_chess()