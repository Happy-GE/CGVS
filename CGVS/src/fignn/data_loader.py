import os
import json
import pandas as pd
import networkx as nx

class RelationalDatabase:
    def __init__(self, tables: dict, foreign_keys: list, primary_keys: dict):#初始化类
        """
        Args:
            tables (dict): table_name -> DataFrame
            foreign_keys (list): (src_table, src_col, dst_table, dst_col)
            primary_keys (dict): table_name -> primary key column
        """
        self.tables = tables
        self.foreign_keys = foreign_keys
        self.primary_keys = primary_keys

    def get_table(self, name):
        return self.tables[name]

    def get_foreign_keys(self):
        return self.foreign_keys

    def get_all_attributes(self, exclude_keys=True, label_table=None, max_depth=None):
        """
        Return a list of (table, attribute) pairs that are promotable.

        If label_table and max_depth are specified, restrict to attributes reachable via FK paths of length ≤ max_depth.

        Args:
            exclude_keys (bool): Exclude PK and FK columns
            label_table (str or None): If specified, only return attributes reachable from this table
            max_depth (int or None): FK path limit from label_table

        Returns:
            List of (table, attribute) pairs
        """
        # Build schema graph if needed
        if not hasattr(self, "schema_graph"):#如果还没建边
            self.schema_graph = nx.Graph()#建一个空图
            for src, _, dst, _ in self.foreign_keys:#遍历该表然后添加边
                self.schema_graph.add_edge(src, dst)

        # Build set of key columns to exclude
        key_columns = set()
        if exclude_keys:
            for table, pk in self.primary_keys.items():
                key_columns.add((table, pk))# 把 (表名, 主键名) 存入黑名单
            for src, src_col, _, _ in self.foreign_keys:
                key_columns.add((src, src_col))#源表名, 外键列名列存下黑名单

        # Determine which tables are reachable from label_table
        if label_table and max_depth is not None:
            # BFS traversal
            reachable_tables = set()#设置一个空集合存储能到达的表
            queue = [(label_table, 0)]# 搜索队列：(当前在哪张表, 距离圆心几步)
            visited = set()# 记录本：防止在两个互相连接的表之间转圈圈

            while queue:
                current, depth = queue.pop(0)# 从队列最前面取出一个“待探测”的表
                if current in visited or depth > max_depth:# 检查：如果这张表已经看过了，或者 步数 已经超过了半径(max_depth)，就跳过
                    continue
                visited.add(current)# 标记：这张表我来过了，并且它是我们要找的“有效表”
                reachable_tables.add(current)# 扩散：在地图(schema_graph)上找这张表的所有邻居
                for neighbor in self.schema_graph.neighbors(current):
                    queue.append((neighbor, depth + 1))
        else:
            reachable_tables = set(self.tables.keys())#程序会认为你想要全家桶。它直接把数据库里所有的表名都塞进

        # Collect promotable attributes
        promotable = []# 第一步：初始化一个空列表，用来存放最后选中的特征
        for table in reachable_tables:# 第二步：第一层循环 —— 遍历刚才“雷达”搜到的所有
            for col in self.tables[table].columns:# 第三步：第二层循环 —— 翻开这张表，看它所有的列（Columns）
                if (table, col) not in key_columns:# 第四步：核心筛选 —— 检查这一列是不是在“黑名单”里# 如果 (表名, 列名) 不在 key_columns（即主键和外键）里
                    promotable.append((table, col))# 第五步：收录 —— 认为这是一个有用的特征，加入结果集

        return promotable

    def get_fk_path(self, source: str, target: str, max_depth: int = 2):
        """
        Find a FK path between `source` and `target` tables in the schema graph,
        constrained by `max_depth`. Returns a list of table names or None.
        """
        if not hasattr(self, "schema_graph"):
            # Build if not yet initialized
            self.schema_graph = nx.Graph()
            for src, _, dst, _ in self.foreign_keys:
                self.schema_graph.add_edge(src, dst)

        if source not in self.schema_graph or target not in self.schema_graph:
            return None

        try:
            path = nx.shortest_path(self.schema_graph, source=source, target=target)
            if len(path) - 1 <= max_depth:
                return path
            return None
        except nx.NetworkXNoPath:
            return None

    def print_schema(self):
        print("Tables:", list(self.tables.keys()))
        print("Foreign Keys:")
        for fk in self.foreign_keys:
            print(f"  {fk[0]}.{fk[1]} → {fk[2]}.{fk[3]}")


def load_relational_data(data_dir: str) -> RelationalDatabase:
    schema_path = os.path.join(data_dir, "schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)#data_dir 指向你的文件夹。程序先读 schema.json

    tables = {
        table_name: pd.read_csv(os.path.join(data_dir, f"{table_name}.csv"))
        for table_name in schema["tables"]
    }

    return RelationalDatabase(
        tables=tables,
        foreign_keys=schema["foreign_keys"],
        primary_keys=schema["primary_keys"]
    )

def load_tabular_data(df: pd.DataFrame, table_name: str = "main", pk: str = None) -> RelationalDatabase:
    """
    Wrap a single flat DataFrame into a RelationalDatabase object.

    Args:
        df (pd.DataFrame): Input single-table data.
        table_name (str): Name to assign to the table in the schema.
        pk (str or None): Name of the primary key column. If None, use the index.

    Returns:
        RelationalDatabase: a minimal wrapper compatible with graph builders.
    """
    if pk is None:
        # Ensure index is named; if not, assign a temp name
        index_name = df.index.name or "__index__"
        df = df.copy()
        df[index_name] = df.index
        pk = index_name

    tables = {table_name: df}
    primary_keys = {table_name: pk}
    foreign_keys = []

    return RelationalDatabase(
        tables=tables,
        foreign_keys=foreign_keys,
        primary_keys=primary_keys
    )


if __name__ == "__main__":
    print("Data loader test")
    db = load_relational_data("data/synthetic")#读取（JSON）的剧本
    db.print_schema()

    print("Available attributes for promotion:")
    print(db.get_all_attributes())

    print("Max depth = 2")
    print(db.get_all_attributes(label_table="dispat", max_depth=2))
