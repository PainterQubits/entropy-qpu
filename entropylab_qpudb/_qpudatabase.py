import json
import base64
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Type, Optional, Dict

import ZODB
import ZODB.FileStorage
from paramdb import ParamDB, CommitEntry
from paramdb._database import _Snapshot, _compress
import pandas as pd
import transaction
from entropylab.instruments.instrument_driver import Resource
from persistent import Persistent
from persistent.mapping import PersistentMapping
from zc.lockfile import LockError

from entropylab_qpudb._resolver import DefaultResolver


class CalState(Enum):
    UNCAL = auto()
    COARSE = auto()
    MED = auto()
    FINE = auto()

    def __str__(self):
        return self.name


@dataclass(repr=True)
class ConfidenceInterval:
    error: float
    confidence_level: float = -1


@dataclass(repr=False)
class QpuParameter(Persistent):
    """
    A QPU parameter which stores values and modification status for QPU DB entries
    """

    value: Any
    last_updated: datetime = None
    cal_state: CalState = CalState.UNCAL
    confidence_interval: ConfidenceInterval = ConfidenceInterval(-1)

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

    def __repr__(self):
        if self.value is None:
            return "QpuParameter(None)"
        else:
            return (
                f"QpuParameter(value={self.value}, "
                f"last updated: {self.last_updated.strftime('%m/%d/%Y %H:%M:%S')}, "
                f"calibration state: {self.cal_state}), "
                f"confidence_interval: {self.confidence_interval})"
            )


@dataclass(repr=False, frozen=True)
class FrozenQpuParameter:
    """
    A QPU parameter which stores values and modification status for QPU DB entries
    """

    value: Any
    last_updated: datetime = None
    cal_state: CalState = CalState.UNCAL
    confidence_interval: ConfidenceInterval = ConfidenceInterval(-1)

    def __repr__(self):
        if self.value is None:
            return "QpuParameter(None)"
        else:
            return (
                f"QpuParameter(value={self.value}, "
                f"last updated: {self.last_updated.strftime('%m/%d/%Y %H:%M:%S')}, "
                f"calibration state: {self.cal_state}), "
                f"confidence_interval: {self.confidence_interval})"
            )


def _db_file_from_path(path, dbname):
    return os.path.join(path, dbname + ".fs")


def _hist_file_from_path(path, dbname, old: bool = False):
    extension = "fs" if old else "db"
    return os.path.join(path, dbname + f"_history.{extension}")


def create_new_qpu_database(
    dbname: str,
    initial_data_dict: Dict = None,
    force_create: bool = False,
    path: str = None,
) -> None:
    """
    Create a new QPU database permanent storage file. This operation is performed once in the lifetime of a database,
    and is quite similar to initializing a git repo.

    When this method is called, DB files are generated in the working directory of the script by default. An attempt
    to create a DB where one already exists by the same name will lead to an error, unless `force_create` flag is used.

    .. note::

        if the scripts that create this database are managed by a version control system such as git, it is not
        recommended to create the DB files in the same path as the scripts, since then they will be tracked by the
        versioning system or, if ignored, can be deleted by it.

    :param dbname: The name of the database. Used when opening with `QpuDatabaseConnection`.
    :param initial_data_dict: An initial dictionary of QPU parameters.
    :param force_create: If set to true, calling this method when an array already exists in the folder will lead to
    it being overridden.
    :param path: The path where the DB is to be stored.
    """
    if initial_data_dict is None:
        initial_data_dict = {}
    if path is None:
        path = os.getcwd()
    dbfilename = _db_file_from_path(path, dbname)
    if os.path.isfile(dbfilename) and not force_create:
        raise FileExistsError(f"db files for {dbname} already exists")
    storage = ZODB.FileStorage.FileStorage(dbfilename)
    db = ZODB.DB(storage)
    connection = db.open()
    root = connection.root()

    # promote all attributes to QpuParams
    # todo: turn into validation schema
    # todo: assert num_qubits is in system
    initial_data_dict = deepcopy(initial_data_dict)
    for element in initial_data_dict.keys():
        for attr in initial_data_dict[element].keys():
            parameter = initial_data_dict[element][attr]
            if not isinstance(parameter, QpuParameter):
                initial_data_dict[element][attr] = QpuParameter(parameter)

    root["elements"] = PersistentMapping(initial_data_dict)
    transaction.commit()
    db.close()

    # create history db
    hist_filename = _hist_file_from_path(path, dbname)
    db_hist = ParamDB[Optional[str]](hist_filename)
    db_hist.commit("Initial commit", None)
    db_hist.dispose()


class ReadOnlyError(Exception):
    pass


class _QpuDatabaseConnectionBase(Resource):
    def connect(self):
        pass

    def teardown(self):
        self.close()

    def revert_to_snapshot(self, snapshot: str):
        pass

    def snapshot(self, update: bool) -> str:
        return json.dumps(
            {
                "qpu_name": self._dbname,
                "index": self._db_hist.num_commits - 1,
                "message": self._db_hist.latest_commit.message,
            }
        )

    @staticmethod
    def deserialize_function(snapshot: str, class_object: Type):
        data = json.loads(snapshot)
        return class_object(data["qpu_name"])

    def __init__(self, dbname, history_index=None, path=None):
        if path is None:
            path = os.getcwd()
        self._path = path
        self._dbname = dbname
        dbfilename = _db_file_from_path(self._path, self._dbname)
        if not os.path.exists(dbfilename):
            raise FileNotFoundError(f"QPU DB {self._dbname} does not exist")
        self._db = None
        super().__init__()
        self._db_hist = self._open_hist_db()
        self._con = self._open_data_db(history_index)

    def _open_data_db(self, history_index):
        dbfilename = _db_file_from_path(self._path, self._dbname)
        assert self._db_hist.num_commits > 0, "history database is empty"
        if history_index is not None:
            commit_entry = self._db_hist.commit_history(
                history_index, history_index + 1
            )[0]
            connected_tx = self._db_hist.load(commit_entry.id)
            at = None if connected_tx is None else base64.b64decode(connected_tx)
        else:
            commit_entry = self._db_hist.latest_commit
            at = None
        try:
            self._db = ZODB.DB(dbfilename) if self._db is None else self._db
        except LockError:
            raise ConnectionError(
                f"attempting to open a connection to {self._dbname} but a connection"
                " already exists. Try closing existing python sessions."
            )

        con = self._db.open(transaction_manager=transaction.TransactionManager(), at=at)
        con.transaction_manager.begin()
        print(
            f"opening qpu database {self._dbname} from commit"
            f" {self._str_hist_entry(commit_entry)} at index {commit_entry.id - 1}"
        )
        return con

    def _open_hist_db(self):
        histfilename = _hist_file_from_path(self._path, self._dbname)
        db_hist_exists = os.path.exists(histfilename)
        db_hist = ParamDB[Optional[str]](histfilename)
        if db_hist_exists:
            return db_hist
        old_histfilename = _hist_file_from_path(self._path, self._dbname, old=True)
        if os.path.exists(old_histfilename):
            # copy the contents of the old history database into the new one
            try:
                old_db_hist = ZODB.DB(old_histfilename, read_only=True)
                # hack so commit timestamps match, based on internal ParamDB.commit()
                # implementation
                with db_hist._Session.begin() as session:
                    for entry in old_db_hist.open().root()["entries"]:
                        connected_tx: Optional[bytes] = entry["connected_tx"]
                        data = (
                            None
                            if connected_tx is None
                            else base64.b64encode(connected_tx).decode()
                        )
                        session.add(
                            _Snapshot(
                                message=entry["message"] or "",
                                timestamp=entry["timestamp"],
                                data=_compress(json.dumps(data)),
                            )
                        )
                return db_hist
            except Exception as exc:
                # remove the new history database if the conversion failed
                db_hist.dispose()
                os.remove(histfilename)
                raise exc
        raise RuntimeError(f"no history database for qpu database {self._dbname}")

    def __enter__(self):
        return self

    @property
    def readonly(self):
        return self._con.isReadOnly()

    def close(self) -> None:
        """
        Closes QPU DB connection to allow for other connections.
        """
        print(f"closing qpu database {self._dbname}")
        self._con._db.close()
        self._db_hist.dispose()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def set(
        self,
        element: str,
        attribute: str,
        value: Any,
        new_cal_state: Optional[CalState] = None,
        new_confidence_interval: Optional[ConfidenceInterval] = None,
    ) -> None:
        """
        A generic function for modifying values of element attributes.

        Modifies the value in memory. The modified value will persist until the connection object
        is deleted. To store the object in memory constantly,
        use :func:`~entropylab_qpudb._qpudatabase.QpuDatabaseConnectionBase.commit`.

        :param element: The name of the element whose attribute should be modified
        :param attribute: The name of the attribute
        :param value: The value to modify
        :param new_cal_state: (optional) new calibration state specification
        :param new_confidence_interval: (optional) a ConfidenceInterval object which holds the error in this parameter
        """
        root = self._con.root()

        if element not in root["elements"]:
            raise AttributeError(
                f"element {element} does not exist for element {element}"
            )
        if attribute not in root["elements"][element]:
            raise AttributeError(
                f"attribute {attribute} does not exist for element {element}"
            )
        if new_cal_state is None:
            new_cal_state = root["elements"][element][attribute].cal_state
        if new_confidence_interval is None:
            new_confidence_interval = root["elements"][element][
                attribute
            ].confidence_interval
        root["elements"][element][attribute].value = value
        root["elements"][element][attribute].last_updated = datetime.now()
        root["elements"][element][attribute].cal_state = new_cal_state
        root["elements"][element][
            attribute
        ].confidence_interval = new_confidence_interval

    def add_attribute(
        self,
        element: str,
        attribute: str,
        value: Any = None,
        new_cal_state: Optional[CalState] = None,
        new_confidence_interval: Optional[ConfidenceInterval] = None,
    ) -> None:
        """
        Adds an attribute to an existing element.

        :raises: AttributeError if attribute already exists.
        :param element: the name of the element to add
        :param attribute: the name of the new atrribute
        :param value: an optional value for the new attribute
        :param new_cal_state: an optional new cal state
        :param new_confidence_interval: (optional) a ConfidenceInterval object which holds the error in this parameter
        """
        root = self._con.root()

        if element not in root["elements"]:
            raise AttributeError(
                f"element {element} does not exist for element {element}"
            )
        if attribute in root["elements"][element]:
            raise AttributeError(
                f"attribute {attribute} already exists for element {element}"
            )
        else:
            root["elements"][element][attribute] = QpuParameter(
                value, datetime.now(), new_cal_state
            )
            if new_confidence_interval is not None:
                root["elements"][element][
                    attribute
                ].confidence_interval = new_confidence_interval
            root["elements"]._p_changed = True

    def remove_attribute(self, element: str, attribute: str) -> None:
        """
        remove an existing attribute.

        :raises: AttributeError if attribute does not exist.
        :param element: the name of the element
        :param attribute: the name of the attribute to remove
        """
        root = self._con.root()

        if element not in root["elements"]:
            raise AttributeError(
                f"element {element} does not exist for element {element}"
            )
        if attribute not in root["elements"][element]:
            raise AttributeError(
                f"attribute {attribute} does not exist for element {element}"
            )
        else:
            del root["elements"][element][attribute]
            root["elements"]._p_changed = True

    def add_element(self, element: str) -> None:
        """
        :raises: AttributeError if element already exists.
        Adds a new element to the DB
        :param element: the name of the element to add
        """
        root = self._con.root()
        if element in root["elements"]:
            raise AttributeError(f"element {element} already exists")
        else:
            root["elements"][element] = dict()

    def get(self, element: str, attribute: str) -> FrozenQpuParameter:
        """
        Get a QpuParameter object from which values, last modified and calibration can be extracted.

        :param element: name of the element from which to get
        :param attribute: name of the attribute to get
        :return: a :class:`entropylab_qpudb._qpudatabase.FrozenQpuParameter` instance from which values and modification
        data can be obtained
        """
        root = self._con.root()
        if element not in root["elements"]:
            raise AttributeError(
                f"element {element} does not exist for element {element}"
            )
        if attribute not in root["elements"][element]:
            raise AttributeError(
                f"attribute {attribute} does not exist for element {element}"
            )
        return FrozenQpuParameter(
            deepcopy(root["elements"][element][attribute].value),
            deepcopy(root["elements"][element][attribute].last_updated),
            deepcopy(root["elements"][element][attribute].cal_state),
            deepcopy(root["elements"][element][attribute].confidence_interval),
        )

    def commit(self, message: Optional[str] = None) -> None:
        """
        Permanently store the existing state to the DB and add a new commit to the history list
        :param message: an optional message for the commit
        """
        if self.readonly:
            raise ReadOnlyError("Attempting to commit to a DB in a readonly state")
        lt_before = self._con._db.lastTransaction()
        self._con.transaction_manager.commit()
        lt_after = self._con._db.lastTransaction()
        if lt_before != lt_after:  # this means a commit actually took place
            commit_id = self._db_hist.commit(
                message or "", base64.b64encode(lt_after).decode()
            )
            message_index = commit_id - 1
            commit_entry = self._db_hist.commit_history(
                message_index, message_index + 1
            )[0]
            print(
                f"commiting qpu database {self._dbname} with commit"
                f" {self._str_hist_entry(commit_entry)} at index {message_index}"
            )
        else:
            print("did not commit")

    def abort(self):
        self._con.transaction_manager.abort()

    def print(self, element=None):
        # todo add resolver and invert it
        if element is None:
            data = self._con.root()["elements"]
            for element in data:
                print("\n" + element + "\n----")
                for attr in data[element]:
                    print(f"{attr}:\t{data[element][attr]}")
        else:
            data = self._con.root()["elements"]
            print("\n" + element + "\n----")
            for attr in data[element]:
                print(f"{attr}:\t{data[element][attr]}")

    def get_history(self) -> pd.DataFrame:
        history = self._db_hist.commit_history()
        return pd.DataFrame(
            [{"timestamp": entry.timestamp, "message": entry.message}]
            for entry in history
        )

    @staticmethod
    def _str_hist_entry(hist_entry: CommitEntry):
        return (
            f"<timestamp: {hist_entry.timestamp.strftime('%m/%d/%Y %H:%M:%S')},"
            f" message: {hist_entry.message}>"
        )

    def restore_from_history(self, history_index: int) -> None:
        """
        restore the current unmodified and open DB data to be the same as the one from `history_index`.
        Will not commit the restored data.

        .. note::

            The `last_modified` values will return to the ones in the stored commit as well.

        :param history_index: History index from which to restore
        """
        con = self._open_data_db(history_index)
        self._con.root()["elements"] = deepcopy(con.root()["elements"])
        con.close()


class QpuDatabaseConnection(_QpuDatabaseConnectionBase):
    def __init__(self, dbname, resolver=None, **kwargs):
        super().__init__(dbname, **kwargs)
        if resolver is None:
            self._resolver = DefaultResolver()
        else:
            self._resolver = resolver

    def q(self, qubit):
        element = self._resolver.q(qubit)
        return QpuAdapter(element, self)

    def res(self, res):
        element = self._resolver.res(res)
        return QpuAdapter(element, self)

    def coupler(self, qubit1, qubit2):
        element = self._resolver.coupler(qubit1, qubit2)
        return QpuAdapter(element, self)

    def system(self):
        return QpuAdapter("system", self)

    def update_q(self, qubit, field, value, new_cal_state=None):
        self.set(self._resolver.q(qubit), field, value, new_cal_state)

    def update_res(self, res, field, value, new_cal_state=None):
        self.set(self._resolver.res(res), field, value, new_cal_state)

    def update_coupler(self, qubit1, qubit2, field, value, new_cal_state=None):
        self.set(self._resolver.coupler(qubit1, qubit2), field, value, new_cal_state)

    def update_system(self, field, value, new_cal_state=None):
        self.set("system", field, value, new_cal_state)

    @property
    def num_qubits(self):
        return self.get("system", "num_qubits").value


class QpuAdapter(object):
    def __init__(self, element, db) -> None:
        self._element = element
        self._db = db

    def __getattr__(self, attribute: str) -> Any:
        return self._db.get(self._element, attribute)
